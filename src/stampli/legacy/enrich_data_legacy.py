
import os
import asyncio
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- Configuration ---
import sys
from pathlib import Path

# Ensure we can import biocontext modules
# SCRIPT_DIR = enrichment/
SCRIPT_DIR = Path(__file__).parent.resolve()
# We want to add 'src' to sys.path
SRC_DIR = SCRIPT_DIR.parents[3] # enrichment -> stampli -> apps -> biocontext -> src (Wait: biocontext is in src. So parents[3] is src?
# enrichment(0) -> stampli(1) -> apps(2) -> biocontext(3) -> src(4)?
# file is in enrichment/
# parents[0] = enrichment
# parents[1] = stampli
# parents[2] = apps
# parents[3] = biocontext
# parents[4] = src
sys.path.append(str(SRC_DIR))

from stampli.paths import REVIEWS_PARQUET, REVIEWS_ENRICHED_V1

# --- Configuration ---
INPUT_FILE = REVIEWS_PARQUET
OUTPUT_FILE = REVIEWS_ENRICHED_V1
BATCH_SIZE = 50
limit_env = os.environ.get("ENRICH_LIMIT")
LIMIT = int(limit_env) if limit_env else None
MODEL_NAME = "gpt-4o-mini"

# --- Schema ---
class ReviewEnrichment(BaseModel):
    sentiment_score: int = Field(..., description="Integer score 1(Negative) to 5(Positive) based strictly on text tone")
    sentiment_label: str = Field(..., description="One word: Positive, Negative, Neutral, Mixed")
    topics: List[str] = Field(..., description="List of max 3 main topics e.g. 'Queues', 'Food', 'Staff', 'Price'")
    is_complaint: bool = Field(..., description="True if the review contains a specific complaint")

# --- Logic ---

async def enrich_batch(texts: List[str], llm):
    # We construct a prompt to process a batch if possible, or single threaded.
    # For extraction quality, processing one by one appearing in a list is cheaper token-wise 
    # but harder to parse if one fails.
    # Let's do simple async gathering for now.
    
    parser = PydanticOutputParser(pydantic_object=ReviewEnrichment)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data enrichment expert. Extract structured data from the review. \n{format_instructions}"),
        ("user", "{review_text}")
    ])
    
    chain = prompt | llm | parser
    
    tasks = [chain.ainvoke({"review_text": text, "format_instructions": parser.get_format_instructions()}) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run conversion notebook first.")
        return

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    if LIMIT:
        print(f"Limiting to first {LIMIT} rows for testing...")
        df = df.head(LIMIT)

    print(f"Processing {len(df)} reviews with {MODEL_NAME}...")
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, max_retries=2)
    
    # New columns
    sentiments = []
    labels = []
    topics = []
    complaints = []
    
    # Process in batches (simulated by chunking dataframe)
    # Async loop is tricky in synchronous script main, so we run local loop
    loop = asyncio.get_event_loop()
    
    # We'll just iterate for simplicity in this script, or use run_until_complete
    # Chunking for progress
    chunks = np.array_split(df, max(1, len(df) // BATCH_SIZE))
    
    for i, chunk in enumerate(chunks):
        print(f"Batch {i+1}/{len(chunks)}...")
        texts = chunk['Review_Text'].astype(str).tolist()
        
        batch_results = loop.run_until_complete(enrich_batch(texts, llm))
        
        for res in batch_results:
            if isinstance(res, ReviewEnrichment):
                sentiments.append(res.sentiment_score)
                labels.append(res.sentiment_label)
                topics.append(res.topics)
                complaints.append(res.is_complaint)
            else:
                print(f"Failed row: {res}")
                sentiments.append(None)
                labels.append("Error")
                topics.append([])
                complaints.append(False)

    df['enriched_sentiment_score'] = sentiments
    df['enriched_sentiment_label'] = labels
    df['enriched_topics'] = topics
    df['enriched_is_complaint'] = complaints
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("Enrichment Complete.")

if __name__ == "__main__":
    main()
