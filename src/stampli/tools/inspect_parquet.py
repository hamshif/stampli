
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path

# Ensure we can import biocontext modules
# SCRIPT_DIR = tools/
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[4] # tools -> stampli -> apps -> biocontext -> src
sys.path.append(str(SRC_DIR))

from stampli.paths import DATA_DIR, REVIEWS_ENRICHED_V1, REVIEWS_ENRICHED_V3_LLAMA

BASE_DIR = DATA_DIR
V1_PATH = REVIEWS_ENRICHED_V1
V3_PATH = REVIEWS_ENRICHED_V3_LLAMA

def inspect(path, label):
    print(f"\n--- {label}: {path} ---")
    if not path.exists():
        print("❌ Path does not exist.")
        return

    try:
        df = pd.read_parquet(path)
        print(f"Total Rows: {len(df)}")
        if "Branch" in df.columns:
            print("Branch Distribution:")
            print(df["Branch"].value_counts())
        else:
            print("❌ 'Branch' column missing.")
            
        if "crowd_level" in df.columns:
            print("\nCrowd Level Distribution:")
            print(df["crowd_level"].value_counts(dropna=False))
        
    except Exception as e:
        print(f"❌ Error reading: {e}")

if __name__ == "__main__":
    inspect(V1_PATH, "V1 (Current)")
    inspect(V3_PATH, "V3 (Llama)")
