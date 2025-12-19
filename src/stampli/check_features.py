
import sys
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession

# Mocking the path setup from the notebook
# Assuming we are running from src/stampli or similar, but let's just make sure stampli is in path
# The user's notebook has: src_path = str(notebook_dir.parents[0])
# Let's try to locate the stampli package.
import os
sys.path.append(os.path.join(os.environ['HOME'], 'tmp/stampli/src'))

from stampli.paths import get_enriched_path

def main():
    enriched_path = str(get_enriched_path())
    print(f"Reading from: {enriched_path}")
    
    # Use pandas to read parquet directly if possible, often easier for quick inspection if file is local
    try:
        df = pd.read_parquet(enriched_path)
        print("Loaded with Pandas")
    except Exception as e:
        print(f"Pandas load failed ({e}), trying Spark...")
        spark = (SparkSession.builder
            .appName("DataCheck")
            .master("local[*]")
            .getOrCreate())
        df = spark.read.parquet(enriched_path).toPandas()
        
    cols_to_check = ['staff_sentiment', 'price_sensitivity', 'crowd_level', 'topics', 'sentiment_score']
    
    print("\n--- Column Info ---")
    print(df[cols_to_check].info())
    
    print("\n--- Sample Data (Non-null) ---")
    for col in cols_to_check:
        if col in df.columns:
            print(f"\nCol: {col}")
            print(df[col].dropna().head(10))
            print(f"Unique values (top 5): {df[col].value_counts().head(5)}")

if __name__ == "__main__":
    main()
