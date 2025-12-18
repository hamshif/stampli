import json
import textwrap
from pathlib import Path

NB_PATH = "/home/gideon/dev/biocontext/biocontext-services/src/stampli/disney_exploration.ipynb"

def add_comparison_cell():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r') as f:
        nb = json.load(f)

    # Define the code for the new cell
    source_code = textwrap.dedent("""
    # --- COMPARISON: OpenAI (V1) vs Llama 3 (V3) ---
    # Goal: See if Llama 3 extracts 'crowd_level' differently/better.
    
    import pandas as pd
    from stampli.paths import REVIEWS_ENRICHED_V1, REVIEWS_ENRICHED_V3_LLAMA
    
    print("Loading datasets...")
    df_v1 = pd.read_parquet(REVIEWS_ENRICHED_V1)
    df_v3 = pd.read_parquet(REVIEWS_ENRICHED_V3_LLAMA)
    
    print(f"V1 Size: {len(df_v1)}")
    print(f"V3 Size: {len(df_v3)}")
    
    # Merge on review_uid
    # We use inner join to compare only shared reviews (likely Hong Kong subset)
    merged = pd.merge(
        df_v1[['review_uid', 'Review_Text', 'crowd_level', 'Branch']],
        df_v3[['review_uid', 'crowd_level']],
        on='review_uid',
        suffixes=('_openai', '_llama'),
        how='inner'
    )
    
    print(f"Shared Rows: {len(merged)}")
    
    # Comparison Logic
    valid_crowd = ["Packed", "Moderate", "Empty"] # Assuming these are the labels of interest
    
    merged['openai_has_label'] = merged['crowd_level_openai'].notna()
    merged['llama_has_label'] = merged['crowd_level_llama'].notna()
    
    print("\\n--- Extraction Rate ---")
    print(merged[['openai_has_label', 'llama_has_label']].mean() * 100)
    
    # Mismatches (Where one found something and the other didn't, or they disagree)
    # 1. New Extractions (OpenAI missed, Llama found)
    new_finds = merged[~merged['openai_has_label'] & merged['llama_has_label']]
    print(f"\\nLlama found crowd level where OpenAI missed: {len(new_finds)}")
    
    if not new_finds.empty:
        print("Sample of Llama's unique wins:")
        display_scrollable_dataframe(new_finds[['Review_Text', 'crowd_level_openai', 'crowd_level_llama']])
        
    # 2. Lost Extractions (OpenAI found, Llama missed)
    lost_finds = merged[merged['openai_has_label'] & ~merged['llama_has_label']]
    print(f"\\nLlama missed crowd level where OpenAI found: {len(lost_finds)}")
    
    # 3. Disagreements (Both found, but different)
    disagreements = merged[
        merged['openai_has_label'] & 
        merged['llama_has_label'] & 
        (merged['crowd_level_openai'] != merged['crowd_level_llama'])
    ]
    print(f"\\nDisagreements: {len(disagreements)}")
    
    if not disagreements.empty:
        print("Sample Disagreements:")
        display_scrollable_dataframe(disagreements[['Review_Text', 'crowd_level_openai', 'crowd_level_llama']])
        
    # Crosstab
    print("\\n--- Confusion Matrix ---")
    print(pd.crosstab(merged['crowd_level_openai'].fillna("None"), merged['crowd_level_llama'].fillna("None")))
    """)

    new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_code.strip().splitlines(keepends=True)
    }
    
    nb['cells'].append(new_cell)
    
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=4)

    print("Appended comparison cell successfully.")

if __name__ == "__main__":
    add_comparison_cell()
