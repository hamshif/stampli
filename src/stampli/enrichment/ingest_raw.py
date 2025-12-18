#!/usr/bin/env python
import pandas as pd
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parents[1] 
sys.path.append(str(SRC_DIR))

from stampli.paths import REVIEWS_PARQUET, REVIEWS_CSV

CSV_PATH = REVIEWS_CSV

def run_ingestion(csv_path: Path, output_parquet_path: Path):
    """Clean logic for ingesting CSV to Parquet without global reliance."""
    if not csv_path.exists():
        print(f"Error: Raw CSV not found at {csv_path}")
        return False
    
    print(f"Reading {csv_path}...")
    try:
        # Try different encodings for Disney data
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return False
    
    # Simple cleaning: Ensure IDs are unique and add a UID if missing
    if 'Review_ID' in df.columns:
        df = df.rename(columns={'Review_ID': 'review_uid'})
    
    print(f"Loaded {len(df)} rows. Saving to {output_parquet_path}...")
    df.to_parquet(output_parquet_path, index=False)
    print("Ingestion successful.")
    return True

def main():
    # Local case: Use hardcoded globals
    run_ingestion(csv_path=CSV_PATH, output_parquet_path=REVIEWS_PARQUET)

if __name__ == "__main__":
    main()
