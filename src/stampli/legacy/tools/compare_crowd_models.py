import pandas as pd
import sys
from pathlib import Path

# Setup paths
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[4]
sys.path.append(str(SRC_DIR))

from stampli.paths import REVIEWS_ENRICHED_V1, REVIEWS_ENRICHED_V3_LLAMA

def check_dataset(name, path):
    print(f"\n--- Analyzing {name} ---")
    if not path.exists():
        print(f"Path {path} does not exist.")
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return

    # Helper for month
    def get_month(ym):
        try:
            return int(ym.split('-')[1])
        except:
            return 0
            
    df['month'] = df['Year_Month'].apply(get_month)
    
    # Filter: Hong Kong (Available in both)
    # V3 (Llama) seems to only have Hong Kong data
    subset = df[df['Branch'] == 'Disneyland_HongKong']
    
    total = len(subset)
    print(f"Total Hong Kong Reviews: {total}")
    
    if subset.empty:
        return

    if 'crowd_level' not in subset.columns:
        print("'crowd_level' column missing!")
        return

    # Captured Distribution
    print("Crowd Labels:")
    print(subset['crowd_level'].value_counts(dropna=False))
    
    # False Negative Analysis
    keywords = ['crowd', 'packed', 'busy', 'line', 'wait', 'queue']
    mask_keys = subset['Review_Text'].str.contains('|'.join(keywords), case=False, na=False)
    
    # "Hit" = Label is NOT None (and presumably captures crowd, but just checking presence first)
    # "Miss" = Keywords present BUT Label is None
    
    missed = subset[mask_keys & subset['crowd_level'].isna()]
    keyword_matches = subset[mask_keys]
    
    miss_rate = (len(missed) / len(keyword_matches) * 100) if not keyword_matches.empty else 0
    
    print(f"Keyword Matches: {len(keyword_matches)}")
    print(f"Rows with keywords but NO Label (Null): {len(missed)}")
    print(f"Miss Rate: {miss_rate:.1f}%")

def compare():
    check_dataset("V1 (GPT-4o-mini / Baseline)", REVIEWS_ENRICHED_V1)
    check_dataset("V3 (Llama-3)", REVIEWS_ENRICHED_V3_LLAMA)

if __name__ == "__main__":
    compare()
