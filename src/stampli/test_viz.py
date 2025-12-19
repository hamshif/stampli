import matplotlib
matplotlib.use('Agg') # Prevent window popup during testing
import pandas as pd

import sys
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parents[0])) # src/stampli -> src

from stampli.paths import get_enriched_path
from stampli.disney_viz import (
    visualize_theme_sentiment,
    visualize_low_rating_drivers,
    visualize_seasonality,
    visualize_country_sentiment,
    visualize_staff_impact,
    visualize_crowd_impact,
    get_evidence_quotes,
    generate_cx_playbook
)





def main():
    print("Running Viz Test for All Insights...")
    path = get_enriched_path()
    df = pd.read_parquet(path)
    
    root_dir = SCRIPT_DIR.parents[2] # src/stampli -> src -> tmp/stampli
    save_dir = root_dir / "data" / "plots"
    
    print(f"Saving plots to {save_dir}")
    
    # 1. Theme Heatmaps
    visualize_theme_sentiment(df, save_dir=str(save_dir), show=False)
    
    # 2. Low Ratings
    visualize_low_rating_drivers(df, save_dir=str(save_dir), show=False)
    
    # 3. Seasonality
    visualize_seasonality(df, save_dir=str(save_dir), show=False)
    
    # 4. Country Expectations
    visualize_country_sentiment(df, save_dir=str(save_dir), show=False)
    
    # 5. Staff Impact
    visualize_staff_impact(df, save_dir=str(save_dir), show=False)
    
    # 6. Crowding Signal
    visualize_crowd_impact(df, save_dir=str(save_dir), show=False)
    
    # 7. Evidence
    get_evidence_quotes(df, {'crowd_level': 'Packed', 'Branch': 'Disneyland_California'})

    # 8. CX Playbook
    print("\nTesting CX Playbook...")
    try:
        playbook = generate_cx_playbook(df, save_dir=str(save_dir))
        print(f"CX Playbook Generated. Rows: {len(playbook)}")
        print(playbook.head())
    except Exception as e:
        print(f"CX Playbook Failed: {e}")


if __name__ == "__main__":
    main()
