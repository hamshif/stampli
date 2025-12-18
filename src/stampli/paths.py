from pathlib import Path
import os

# This file is at src/stampli/paths.py
PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parents[1] # src/stampli -> src -> PROJECT_ROOT

# Data Directory (stampli/data)
DATA_DIR = PROJECT_ROOT / "data"

# Source CSV
REVIEWS_CSV = DATA_DIR / "DisneylandReviews.csv"

# Parquet Outputs
REVIEWS_PARQUET = DATA_DIR / "reviews.parquet" 
REVIEWS_BASE_PARQUET = REVIEWS_PARQUET # Aliasing since it is a single file now

# Versioning
CURRENT_VERSION = "v4"

def get_enriched_path(version: str = CURRENT_VERSION) -> Path:
    """Returns the path for a given enrichment version."""
    return DATA_DIR / f"reviews_enriched_{version}.parquet"

# Standard shortcut for the current production data
REVIEWS_ENRICHED = get_enriched_path()
