#!/usr/bin/env python


import os
import asyncio
import pandas as pd
import numpy as np
import sys
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parents[1] # enrichment -> src
sys.path.append(str(SRC_DIR))
from stampli.paths import REVIEWS_PARQUET, DATA_DIR, get_enriched_path

# --- Configuration ---
INPUT_FILE = REVIEWS_PARQUET
OUTPUT_FILE = get_enriched_path() # Defaults to centralized version in paths.py
BATCH_SIZE = 50
MODEL_NAME = "gpt-4o-mini"
LIMIT_BATCHES = 3 #None # Default to full run. Pipeline script can override if needed.

# --- Improved Schema ---
class ReviewEnrichment(BaseModel):
    sentiment_score: int = Field(..., description="1 (Negative) to 5 (Positive)")
    sentiment_label: str = Field(..., description="Positive, Negative, Neutral, Mixed")
    topics: List[str] = Field(..., description="Max 3 topics e.g. 'Queues', 'Food', 'Staff'")
    is_complaint: bool = Field(..., description="True if complaint present")
    
    # New Fields with Explicit Options
    crowd_level: Optional[str] = Field(None, description="Inferred crowd density: 'Empty', 'Moderate', 'Packed', 'Crowded'. Default None if not mentioned.")
    staff_sentiment: Optional[str] = Field(None, description="Staff interaction quality: 'Friendly', 'Rude', 'Neutral', 'Helpful'.")
    price_sensitivity: Optional[str] = Field(None, description="Price perception: 'Expensive', 'Worth it', 'Cheap', 'Rip-off'.")
    family_sentiment: Optional[str] = Field(None, description="Family experience: 'Joyful', 'Stressful', 'Tired', 'Harmonious'.")

async def enrich_batch(texts: List[str], llm):
    parser = PydanticOutputParser(pydantic_object=ReviewEnrichment)
    
    # --- PROMPT IMPROVEMENT ---
    # CoT / Explicit Instructions
    system_prompt = """You are a Disney Parks analyst. Extract structured data.
    
    GUIDELINES:
    1. INFER `crowd_level` aggressively. 
       - "Long lines", "hour wait", "people everywhere" -> 'Packed'
       - "Walked on rides", "no wait" -> 'Empty'
    2. INFER `staff_sentiment`. "Cast members were great" -> 'Friendly'. "Rude employee" -> 'Rude'.
    3. INFER `price_sensitivity`. "Overpriced burger" -> 'Expensive'.
    
    If inference is impossible, return null.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{review_text}")
    ])
    
    chain = prompt | llm | parser
    
    # Semaphore or simple list
    tasks = [chain.ainvoke({"review_text": text, "format_instructions": parser.get_format_instructions()}) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

def run_enrichment(
    input_file: Path, 
    output_file: Path, 
    model_name: str = "gpt-4o-mini",
    batch_size: int = 50,
    limit_batches: Optional[int] = None
):
    """Clean logic for LLM enrichment without global reliance."""
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # 1. Load BASE Data
    print(f"\n[PIPELINE] Loading raw data: {input_file}")
    df_main = pd.read_parquet(input_file).copy()
    total_rows = len(df_main)
    
    # Ensure target columns exist
    enrichment_cols = list(ReviewEnrichment.__annotations__.keys())
    for col in enrichment_cols:
        if col not in df_main.columns:
            df_main[col] = None

    # Determine unique key
    if 'review_uid' in df_main.columns:
        key = 'review_uid'
    elif 'Review_ID' in df_main.columns:
        key = 'Review_ID'
    else:
        print("Error: No unique ID found.")
        return

    # 2. Merge Existing Progress (Resume Support)
    if output_file.exists():
        print(f"[PIPELINE] Found checkpoint: {output_file}")
        df_existing = pd.read_parquet(output_file)
        
        # Identify rows that already have LLM enrichment
        # We check both sentiment_score and sentiment_label as a proxy for 'complete' LLM work
        if 'sentiment_score' in df_existing.columns:
            existing_filled = df_existing[df_existing['sentiment_score'].notna()]
        else:
            existing_filled = pd.DataFrame(columns=[key])

        if not existing_filled.empty:
            print(f"[PIPELINE] Recovered {len(existing_filled)} enriched rows from checkpoint.")
            for col in enrichment_cols:
                if col in existing_filled.columns:
                    # Only fill NaNs in the main DF with values from the checkpoint
                    mapping = dict(zip(existing_filled[key], existing_filled[col]))
                    df_main[col] = df_main[col].fillna(df_main[key].map(mapping))

    # 3. Filter for work remaining
    df_remaining = df_main[df_main['sentiment_score'].isna()].copy()
    skipped = total_rows - len(df_remaining)
    
    print(f"[PIPELINE] Data Status: {total_rows} total rows.")
    if skipped > 0:
        print(f"[PIPELINE] Skipping {skipped} rows already enriched.")
    print(f"[PIPELINE] Work remaining: {len(df_remaining)} rows.")

    if len(df_remaining) == 0:
        print("[PIPELINE] All rows are already enriched. Dataset complete.")
        # Ensure the output exists
        df_main.to_parquet(output_file, index=False)
        return

    # 4. LLM Configuration
    print(f"[PIPELINE] Starting Enrichment with {model_name}...")
    llm = ChatOpenAI(model=model_name, temperature=0, max_retries=2)
    loop = asyncio.get_event_loop()
    
    # Process in Batches
    n_remaining = len(df_remaining)
    n_batches = max(1, n_remaining // batch_size)
    chunks = np.array_split(df_remaining, n_batches)
    
    if limit_batches is not None:
        print(f"[DEBUG] Applying limit: {limit_batches} batches.")
        chunks = chunks[:limit_batches]
    
    actual_batches = len(chunks)
    print(f"[PIPELINE] Processing {actual_batches} batches (Size: ~{batch_size})...")

    processed_in_this_run = 0
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0: continue
        
        current_global_start = skipped + processed_in_this_run
        current_global_end = current_global_start + len(chunk)
        print(f"Batch {i+1}/{actual_batches} | Global Progress: {current_global_end}/{total_rows} rows | (Batch Size: {len(chunk)})")
        texts = chunk['Review_Text'].astype(str).tolist()
        
        batch_results = loop.run_until_complete(enrich_batch(texts, llm))
        
        # Parse Results
        chunk_data = {k: [] for k in enrichment_cols}
        for res in batch_results:
            if isinstance(res, ReviewEnrichment):
                for k in chunk_data:
                    chunk_data[k].append(getattr(res, k))
            else:
                for k in chunk_data:
                    chunk_data[k].append(None)
                    
        # Update Main Dataframe
        for col in enrichment_cols:
            mapping = dict(zip(chunk[key], chunk_data[col]))
            df_main.loc[df_main[key].isin(chunk[key]), col] = df_main[key].map(mapping)
            
        # Immediate Save
        df_main.to_parquet(output_file, index=False)
        processed_in_this_run += len(chunk)

    print(f"[PIPELINE] Enrichment stage complete. Saved to {output_file}")

def main():
    # Local case: Use hardcoded globals
    run_enrichment(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        limit_batches=LIMIT_BATCHES
    )

if __name__ == "__main__":
    main()
