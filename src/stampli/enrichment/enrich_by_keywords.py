#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import re

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(SRC_DIR))
from stampli.paths import DATA_DIR, get_enriched_path

# Default output if run alone
DEFAULT_OUTPUT = get_enriched_path()

# --- Keywords ---
KEYWORDS = {
    "crowd_level": {
        "Packed": [r"\bpacked\b", r"\bcrowded\b", r"\bbusy\b", r"\blong lines\b", r"\bqueue\b", r"\bwait time\b", r"\bpeople everywhere\b"],
        "Empty": [r"\bempty\b", r"\bno wait\b", r"\bwalk on\b", r"\bquiet\b", r"\bdeserted\b"],
        "Moderate": [r"\bmoderate\b", r"\bokay wait\b"]
    },
    "staff_sentiment": {
        "Rude": [r"\brude\b", r"\bunhelpful\b", r"\bmean\b", r"\battitude\b", r"\bignored\b"],
        "Friendly": [r"\bfriendly\b", r"\bhelpful\b", r"\bnice\b", r"\bsmiling\b", r"\bgreat staff\b"]
    },
    "price_sensitivity": {
        "Expensive": [r"\bexpensive\b", r"\bpricey\b", r"\bcostly\b", r"\brip off\b", r"\boverpriced\b"],
        "Worth it": [r"\bworth it\b", r"\bvalue\b", r"\breasonable\b"]
    }
}

def apply_keywords(row, field, mapping):
    current = row.get(field)
    if pd.notna(current) and current != "None" and current != None: 
        return current
        
    text = str(row.get("Review_Text", "")).lower()
    
    for label, patterns in mapping.items():
        for pat in patterns:
            if re.search(pat, text):
                return label
    return None

def run_backfill(input_file: Path, output_file: Path):
    """Clean logic for keyword backfill without global reliance. 
    Preserves existing enrichment data if output_file already exists.
    """
    print(f"[PIPELINE] Running Keyword Backfill...")
    
    # 1. Load Data
    # prioritizing existing progress so we don't wipe out LLM work
    if output_file.exists():
        print(f"[PIPELINE] Loading existing data from {output_file}")
        df = pd.read_parquet(output_file)
    else:
        print(f"[PIPELINE] No existing data. Starting fresh from {input_file}")
        if not input_file.exists():
            print(f"Error: Source file missing at {input_file}")
            return
        df = pd.read_parquet(input_file)
        
    print(f"[PIPELINE] Data Status: {len(df)} rows.")
    
    fields_to_process = ["crowd_level", "staff_sentiment", "price_sensitivity"]
    
    stats = {}
    
    for field in fields_to_process:
        if field not in df.columns:
            df[field] = None
            
        initial_filled = df[field].notna().sum()
        
        # Apply Logic
        df[field] = df.apply(lambda row: apply_keywords(row, field, KEYWORDS[field]), axis=1)
        
        final_filled = df[field].notna().sum()
        stats[field] = final_filled - initial_filled
        
    print("\n--- Backfill Stats ---")
    for f, count in stats.items():
        print(f"{f}: Filled {count} new values.")
        
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Backfill Complete.")

def main():
    # Local case: Use versioned defaults
    run_backfill(input_file=get_enriched_path(), output_file=get_enriched_path())

if __name__ == "__main__":
    main()
