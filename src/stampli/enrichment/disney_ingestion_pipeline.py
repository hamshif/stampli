#!/usr/bin/env python
import sys
from pathlib import Path

# Setup paths to ensure we can import our scripts
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
SRC_ROOT = SCRIPT_DIR.parents[1] # enrichment -> src
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Import DAG nodes
import ingest_raw
import enrich_data
import enrich_by_keywords
import derive_features

from stampli.paths import REVIEWS_CSV, REVIEWS_PARQUET, get_enriched_path

# --- CONFIGURATION ---
VERSION = "v4"
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 50
LIMIT_BATCHES = None # For debugging set to 1, None for full run
# ---------------------

def main():
    print("============================================")
    print(f"STAMPLI PIPELINE: {VERSION}")
    print(f"Model: {MODEL_NAME} | Batch Size: {BATCH_SIZE} | Limit: {LIMIT_BATCHES}")
    print("============================================\n")

    # Define Intermediate Paths
    # We assume data dir is where REVIEWS_PARQUET lives
    DATA_DIR = REVIEWS_PARQUET.parent
    
    # Step 0: Raw Ingestion (Optional Check)
    # Source: REVIEWS_CSV -> REVIEWS_PARQUET (Raw Source)
    # We can assume reviews.parquet is stable raw, or re-ingest if needed. 
    # Let's verify raw exists.
    RAW_SOURCE = REVIEWS_PARQUET
    if not RAW_SOURCE.exists():
        print(">>> Step 0: Raw Ingestion")
        ingest_raw.run_ingestion(REVIEWS_CSV, RAW_SOURCE)
    else:
        print(f">>> Step 0: Using existing Raw Source: {RAW_SOURCE}")

    # Step 1: Keyword Backfill -> Intermediate Stage 1
    # Keyword enrichment is fast and cheap.
    STAGE1_FILE = DATA_DIR / f"reviews_stage1_kw_{VERSION}.parquet"
    print(f"\n>>> STEP 1: Keyword Backfill -> {STAGE1_FILE.name}")
    enrich_by_keywords.run_backfill(
        input_file=RAW_SOURCE,
        output_file=STAGE1_FILE
    )

    # Step 2: Feature Derivation -> Intermediate Stage 2
    # Seasonality, column ordering.
    STAGE2_FILE = DATA_DIR / f"reviews_stage2_derived_{VERSION}.parquet"
    print(f"\n>>> STEP 2: Feature Derivation -> {STAGE2_FILE.name}")
    derive_features.run_derivation(
        input_file=STAGE1_FILE,
        output_file=STAGE2_FILE
    )

    # Step 3: LLM Enrichment -> Final Enriched
    # This is the heavy lifting.
    STAGE3_FILE = DATA_DIR / f"reviews_enriched_{VERSION}.parquet"
    print(f"\n>>> STEP 3: LLM Enrichment -> {STAGE3_FILE.name}")
    enrich_data.run_enrichment(
        input_file=STAGE2_FILE, # Use the derived file as input base
        output_file=STAGE3_FILE,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        limit_batches=LIMIT_BATCHES
    )

    print("\n============================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final Artifact: {STAGE3_FILE}")
    print("============================================")

if __name__ == "__main__":
    main()
