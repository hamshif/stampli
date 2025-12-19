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

# --- CONFIGURATION ---
# All pipeline settings centralized here
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 50
LIMIT_BATCHES = None # For debugging set to 1, None for full run
# ---------------------

from stampli.paths import REVIEWS_CSV, REVIEWS_PARQUET, get_enriched_path

def main():
    print("============================================")
    print("STAMPLI PIPELINE: STARTING")
    print(f"Model: {MODEL_NAME} | Batch Size: {BATCH_SIZE} | Limit: {LIMIT_BATCHES}")
    print("============================================\n")

    # Step 1: Ingestion
    print(">>> STEP 1: Ingestion")
    if not ingest_raw.run_ingestion(
        csv_path=REVIEWS_CSV, 
        output_parquet_path=REVIEWS_PARQUET
    ):
        print("Pipeline aborted at Step 1.")
        return

    # Step 2: Keyword Backfill (Instant results for all rows)
    print("\n>>> STEP 2: Keyword Backfill")
    # This provides a safety net of tags for the entire 42k dataset immediately
    enrich_by_keywords.run_backfill(
        input_file=REVIEWS_PARQUET,
        output_file=get_enriched_path()
    )

    # Step 2.5: Feature Derivation (Season, Clean Cols)
    print("\n>>> STEP 2.5: Feature Derivation")
    derive_features.run_derivation(
        input_file=get_enriched_path(),
        output_file=get_enriched_path()
    )

    # Step 3: LLM Enrichment (Nuanced refinement)
    print("\n>>> STEP 3: LLM Enrichment")
    # This resumes from the file Keywords just created, 
    # and fills in the gaps or refinements.
    enrich_data.run_enrichment(
        input_file=REVIEWS_PARQUET,
        output_file=get_enriched_path(),
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        limit_batches=LIMIT_BATCHES
    )

    print("\n============================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final Data: {get_enriched_path()}")
    print("============================================")

if __name__ == "__main__":
    main()
