
import pandas as pd
import sys
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parents[0])) # src/stampli -> src

from stampli.paths import get_enriched_path
from stampli.disney_viz import analyze_insight_1

def main():
    print("Running Viz Test...")
    path = get_enriched_path()
    df = pd.read_parquet(path)
    
    # Run analysis with save option
    # Need to go up from src/stampli into root/data/plots? 
    # Current location: src/stampli/test_viz.py
    # Root is src/stampli/../../
    
    # Just use absolute path for safety in test
    root_dir = SCRIPT_DIR.parents[2] # src/stampli -> src -> tmp/stampli
    save_dir = root_dir / "data" / "plots"
    
    analyze_insight_1(df, save_dir=str(save_dir))

if __name__ == "__main__":
    main()
