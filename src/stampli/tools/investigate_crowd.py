import pandas as pd
import sys
from pathlib import Path

# Setup paths
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[4]
sys.path.append(str(SRC_DIR))

from stampli.paths import REVIEWS_ENRICHED_V1

def investigate():
    print(f"Loading {REVIEWS_ENRICHED_V1}...")
    df = pd.read_parquet(REVIEWS_ENRICHED_V1)
    
    # helper for month
    def get_month(ym):
        try:
            return int(ym.split('-')[1])
        except:
            return 0
            
    df['month'] = df['Year_Month'].apply(get_month)
    
    # Filter: California + June
    subset = df[
        (df['Branch'] == 'Disneyland_California') & 
        (df['month'] == 6)
    ]
    
    print(f"\n--- Stats for Disneyland_California in June ---")
    print(f"Total Reviews: {len(subset)}")
    
    if subset.empty:
        print("No reviews found for this criteria.")
        return

    # Check crowd_level distribution
    if 'crowd_level' in subset.columns:
        print("\nCrowd Level Distribution:")
        print(subset['crowd_level'].value_counts(dropna=False))
    else:
        print("\n'crowd_level' column missing!")

    # Keyword Search text vs Label
    keywords = ['crowd', 'packed', 'busy', 'line', 'wait', 'queue']
    
    def has_keyword(text):
        if not isinstance(text, str): return False
        return any(k in text.lower() for k in keywords)

    subset_with_keywords = subset[subset['Review_Text'].apply(has_keyword)]
    print(f"\nReviews identifying keywords ({keywords}): {len(subset_with_keywords)}")
    
    # Cross Check: Keyword Present BUT Label Missing/Incorrect
    # Assuming 'Packed' or 'Moderate' captures crowding.
    # Where keyword is present but crowd_level is None or 'Empty' (unlikely empty if 'crowd' mentioned?)
    
    missed = subset_with_keywords[subset_with_keywords['crowd_level'].isna()]
    print(f"False Negative Candidates (Keywords present, Label is None): {len(missed)}")
    
    if not missed.empty:
        print("\nSample Missed Rows:")
        for _, row in missed.head(5).iterrows():
            print(f"- [Label: {row.get('crowd_level')}] {row['Review_Text'][:200]}...")

if __name__ == "__main__":
    investigate()
