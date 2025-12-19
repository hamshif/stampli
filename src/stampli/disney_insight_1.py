
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# --- Path Setup ---
# Add src to sys.path so we can import stampli package
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parents[1] # src/stampli -> src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from stampli.paths import get_enriched_path

def get_themes(row):
    themes = set()
    
    # 1. Topics (Normalize)
    # Check if topics is a list or string representation of list
    topics = row.get('topics')
    if isinstance(topics, np.ndarray):
        topics = topics.tolist()
        
    if isinstance(topics, list):
        for t in topics:
            t_lower = str(t).lower()
            if 'queue' in t_lower or 'crowd' in t_lower or 'wait' in t_lower:
                themes.add('Queue/Crowd')
            elif 'food' in t_lower or 'dining' in t_lower:
                themes.add('Food')
            elif 'staff' in t_lower or 'service' in t_lower:
                themes.add('Staff')
            elif 'price' in t_lower or 'cost' in t_lower or 'expensive' in t_lower:
                themes.add('Price')
            elif 'ride' in t_lower or 'attraction' in t_lower:
                themes.add('Rides')
            elif 'clean' in t_lower:
                themes.add('Cleanliness')
            elif 'family' in t_lower or 'kid' in t_lower:
                themes.add('Family')
            elif 'weather' in t_lower:
                themes.add('Weather')
                
    # 2. Crowd Level -> Queue/Crowd
    cl = str(row.get('crowd_level', '')).lower()
    if 'packed' in cl or 'crowded' in cl:
        themes.add('Queue/Crowd')
        
    # 3. Price Sensitivity -> Price
    ps = str(row.get('price_sensitivity', '')).lower()
    if 'expensive' in ps or 'rip' in ps or 'costly' in ps:
        themes.add('Price')
        
    # 4. Staff Sentiment -> Staff (Review likely mentions staff)
    ss = str(row.get('staff_sentiment', ''))
    if ss and ss.lower() != 'none' and ss.lower() != 'nan':
         themes.add('Staff')
         
    return list(themes)

def main():
    print("--- Disney Insight 1 Analysis ---")
    
    path = get_enriched_path()
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows.")
    
    # Apply Theme Extraction
    print("Extracting Themes...")
    df['extracted_themes'] = df.apply(get_themes, axis=1)
    
    # Explode
    df_exploded = df.explode('extracted_themes')
    df_exploded = df_exploded[df_exploded['extracted_themes'].notna()]
    print(f"Exploded rows: {len(df_exploded)}")
    
    # Check Season
    if 'Season' not in df_exploded.columns:
        print("CRITICAL: 'Season' column missing. Please run 'derive_features.py' pipeline stage.")
        return

    # Aggregate
    print("Aggregating Metrics...")
    heatmap_data = (
        df_exploded
        .groupby(['Branch', 'Season', 'extracted_themes'])
        .agg(
            mean_sentiment=('sentiment_score', 'mean'),
            median_sentiment=('sentiment_score', 'median'),
            count=('review_uid', 'count'),
            neg_count=('sentiment_label', lambda x: (x=='Negative').sum())
        )
        .reset_index()
    )
    
    # Share Negative
    heatmap_data['share_negative'] = heatmap_data['neg_count'] / heatmap_data['count']
    
    # Significance Gating (Set to NaN if < 30)
    # create a copy for plotting to avoid changing data for insights? 
    # Insights should also ignore likely
    plot_data = heatmap_data.copy()
    mask_low = plot_data['count'] < 30
    plot_data.loc[mask_low, ['mean_sentiment', 'median_sentiment', 'share_negative']] = np.nan
    
    SEASONS_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    branches = plot_data['Branch'].unique()
    
    for branch in branches:
        branch_data = plot_data[plot_data['Branch'] == branch]
        
        # Pivot
        piv_sent = branch_data.pivot(index='extracted_themes', columns='Season', values='median_sentiment')
        piv_vol = branch_data.pivot(index='extracted_themes', columns='Season', values='count')
        
        # Reindex
        for col in SEASONS_ORDER:
            if col not in piv_sent.columns: piv_sent[col] = np.nan
            if col not in piv_vol.columns: piv_vol[col] = np.nan
            
        piv_sent = piv_sent[SEASONS_ORDER]
        piv_vol = piv_vol[SEASONS_ORDER]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(piv_sent, annot=True, fmt=".1f", cmap="RdYlGn", vmin=1, vmax=5, ax=axes[0])
        axes[0].set_title(f"{branch} - Median Sentiment (1-5)")
        
        sns.heatmap(piv_vol, annot=True, fmt=".0f", cmap="Blues", ax=axes[1])
        axes[1].set_title(f"{branch} - Volume (Count)")
        
        plt.tight_layout()
        filename = f"insight1_{branch}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()
        
        # Insights
        print(f"\n--- Insights for {branch} ---")
        
        # Use ungated data? no, use gated data for robust insights
        valid_cells = branch_data[branch_data['count'] >= 30]
        
        if not valid_cells.empty:
            # 1. Worst Theme
            worst_idx = valid_cells['median_sentiment'].idxmin()
            worst = valid_cells.loc[worst_idx]
            print(f"• CRITICAL ISSUE: '{worst.extracted_themes}' in {worst.Season} has lowest sentiment ({worst.median_sentiment:.1f}, N={worst['count']}).")
            
            # Action
            suggestions = {
                "Queue/Crowd": "Prioritize queue management and capacity limits.",
                "Price": "Review pricing strategy or add value bundles.",
                "Food": "Refresh menu options and improve dining capacity.",
                "Staff": "Invest in cast member training.",
                "Weather": "Increase shaded areas and indoor activities.",
                "Cleanliness": "Increase frequency of cleaning shifts."
            }
            sugg = suggestions.get(worst.extracted_themes, "Investigate root causes.")
            print(f"• ACTION: {sugg}")
            
            # 2. Largest Drop (vs Theme Average? or Branch Average?)
            # Let's compare this worst cell vs the Branch Average for that Theme
            theme_avg = valid_cells[valid_cells['extracted_themes'] == worst.extracted_themes]['median_sentiment'].mean()
            diff = theme_avg - worst.median_sentiment
            if diff > 0.3:
                 print(f"• DROP: This is {diff:.1f} points lower than the {worst.extracted_themes} average ({theme_avg:.1f}) at this park.")
        else:
            print("• No themes met the significance threshold (30 reviews).")
            
        print("-" * 30)

if __name__ == "__main__":
    main()
