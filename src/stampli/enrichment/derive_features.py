#!/usr/bin/env python
import pandas as pd
import sys
from pathlib import Path

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(SRC_DIR))
from stampli.paths import get_enriched_path

# --- Constants ---

# Season Mapping (Venue -> Season -> Months)
# Ported from stampli_chat.py
SEASON_MAP = {
  "Disneyland_HongKong": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disneyland_California": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disneyland_Paris": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disney_World_Florida": { 
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  }
}

# Explicit Column Order
TARGET_START = [
    'Branch', 'Rating', 'sentiment_score', 'sentiment_label',
    'is_complaint', 'crowd_level', 'staff_sentiment', 'price_sensitivity',
    'family_sentiment', 'Reviewer_Location', 'Season'
]

TARGET_END = [
    'topics', 'Review_Text', 'Year_Month', 'review_uid'
]

def get_season(branch, year_month):
    """
    Derives season from Branch and Year_Month (YYYY-M).
    Returns 'Winter', 'Spring', 'Summer', 'Fall' or None.
    """
    if not isinstance(branch, str) or not isinstance(year_month, str):
        return None
        
    # extract month
    try:
        month_str = year_month.split('-')[1]
        month = int(month_str)
    except (IndexError, ValueError):
        return None
        
    # find branch mapping
    branch_map = None
    for k in SEASON_MAP.keys():
        if branch.lower() in k.lower(): # fuzzy match logic similar to stampli_chat
            branch_map = SEASON_MAP[k]
            break
            
    if not branch_map:
        return None
        
    # lookup season
    for season, months in branch_map.items():
        if month in months:
            # Normalize Fall/Autumn if needed, but map uses 'Autumn'
            return season
            
    return None

def run_derivation(input_file: Path, output_file: Path):
    """
    Derives deterministic features (Season) and enforces column ordering.
    """
    print(f"[DERIVE] Starting Feature Derivation on {input_file}...")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    df = pd.read_parquet(input_file)
    print(f"[DERIVE] Loaded {len(df)} rows.")

    # 1. Derive Season
    print("[DERIVE] Computing 'Season'...")
    # Ensure Branch and Year_Month exist
    if 'Branch' in df.columns and 'Year_Month' in df.columns:
        df['Season'] = df.apply(
            lambda row: get_season(row['Branch'], row['Year_Month']), axis=1
        )
    else:
        print("Warning: Missing 'Branch' or 'Year_Month' columns. Skipping Season derivation.")
        df['Season'] = None

    # 2. Reorder Columns
    print("[DERIVE] Enforcing Column Order...")
    
    current_cols = df.columns.tolist()
    
    # Identify extra columns (those not in TARGET_START or TARGET_END)
    # We want to preserve them and place them BEFORE topics (which is start of TARGET_END)
    
    known_cols = set(TARGET_START + TARGET_END)
    extra_cols = [c for c in current_cols if c not in known_cols]
    
    final_order = []
    
    # Add Start
    for c in TARGET_START:
        if c in df.columns:
            final_order.append(c)
        else:
            # If missing, we can arguably skip or add as None. 
            # The user request implies "make sure these are the order", so usually implies existence.
            # But let's be safe: if it doesn't exist, we don't add it (unless we want to create it empty?)
            # User said: "make sure any columns not in the list get added before topics"
            pass

    # Add Extra
    final_order.extend(extra_cols)
    
    # Add End
    for c in TARGET_END:
        if c in df.columns:
            final_order.append(c)
            
    # Check for any columns that might have been missed in the sets for some reason?
    # The logic above covers (Start U End U Extra) = All, since Extra = All - (Start U End).
    
    # Verify we didn't lose anything
    missing = set(current_cols) - set(final_order)
    if missing:
        print(f"Warning: Logic error, missing cols: {missing}")
        final_order.extend(list(missing)) # Fail safe
        
    # Apply order
    df = df[final_order]
    
    print(f"[DERIVE] Final Column Order: {final_order}")
    print(f"[DERIVE] Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("[DERIVE] Complete.")

def main():
    # Local run default
    path = get_enriched_path()
    run_derivation(input_file=path, output_file=path)

if __name__ == "__main__":
    main()
