import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[4]))
from stampli.paths import REVIEWS_ENRICHED_V3_LLAMA

def inspect_v3():
    print(f"Loading {REVIEWS_ENRICHED_V3_LLAMA}...")
    try:
        df = pd.read_parquet(REVIEWS_ENRICHED_V3_LLAMA)
        print(f"Total Rows: {len(df)}")
        print("Columns:", df.columns.tolist())
        if 'Branch' in df.columns:
            print("Branches:", df['Branch'].unique())
            print("Branch Counts:\n", df['Branch'].value_counts())
        
        if 'Year_Month' in df.columns:
             print("Year_Month Sample:", df['Year_Month'].head().tolist())
             
        # Check crowd_level presence
        if 'crowd_level' in df.columns:
             print("Crowd Level Counts:\n", df['crowd_level'].value_counts(dropna=False))
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_v3()
