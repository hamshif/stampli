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
    visualize_crowd_impact,
    get_evidence_quotes,
    generate_cx_playbook
)
from stampli.reporting import generate_excel_playbook, generate_pdf_report





def main():
    print("Running Viz Test for All Insights...")
    path = get_enriched_path()
    df = pd.read_parquet(path)
    
    root_dir = SCRIPT_DIR.parents[2]
    # Setup Paths - dev test output
    root_dir = Path(".").resolve()
    # Use output/disney_exploration as requested
    save_dir = root_dir / "output" / "disney_exploration" 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Viz Test for All Insights...")
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
        print(f"CX Playbook Generated. Rows: {len(playbook)}")
        print(playbook.head())
        
        # 9. Reports
        print("\nTesting Report Generation...")
        generate_excel_playbook(playbook, str(save_dir / "disney_cx_playbook.xlsx"))
        generate_pdf_report(str(save_dir), str(save_dir / "stampli_insights_report.pdf"))
        
    except Exception as e:
        print(f"CX Playbook/Report Failed: {e}")


if __name__ == "__main__":
    main()
