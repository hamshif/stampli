
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import warnings

# Suppress all warnings (Seaborn deprecations etc.)
warnings.filterwarnings("ignore")

# --- Helper Logic ---
def get_themes(row):
    themes = set()
    
    # 1. Topics (Normalize)
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
        
    # 4. Staff Sentiment -> Staff
    ss = str(row.get('staff_sentiment', ''))
    if ss and ss.lower() != 'none' and ss.lower() != 'nan':
         themes.add('Staff')
         
    return list(themes)

def style_title(ax, branch, subject):
    """
    Applies the requested 2-row, Bold, Large title format.
    Replaces underscores in Branch name with spaces.
    """
    branch_clean = branch.replace('_', ' ')
    title_text = f"{branch_clean}\n{subject}"
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)

def save_plot(filename, save_dir):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Saved: {full_path}")

# --- Insight 1: Theme Heatmaps ---
def analyze_insight_1(df, save_dir=None):
    print("\n=== Insight 1: Theme x Sentiment Heatmaps ===")
    
    work_df = df.copy()
    if 'Season' not in work_df.columns:
        print("CRITICAL: 'Season' column missing.")
        return

    work_df['extracted_themes'] = work_df.apply(get_themes, axis=1)
    df_exploded = work_df.explode('extracted_themes')
    df_exploded = df_exploded[df_exploded['extracted_themes'].notna()]
    
    # Aggregation
    heatmap_data = (
        df_exploded.groupby(['Branch', 'Season', 'extracted_themes'])
        .agg(
            median_sentiment=('sentiment_score', 'median'),
            count=('review_uid', 'count')
        ).reset_index()
    )
    
    # Gating
    heatmap_data.loc[heatmap_data['count'] < 30, ['median_sentiment']] = np.nan

    SEASONS_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
    branches = heatmap_data['Branch'].unique()

    for branch in branches:
        branch_data = heatmap_data[heatmap_data['Branch'] == branch]
        piv_sent = branch_data.pivot(index='extracted_themes', columns='Season', values='median_sentiment')
        piv_vol = branch_data.pivot(index='extracted_themes', columns='Season', values='count')
        
        # Reindex
        for col in SEASONS_ORDER:
            if col not in piv_sent.columns: piv_sent[col] = np.nan
            if col not in piv_vol.columns: piv_vol[col] = np.nan
        piv_sent = piv_sent[SEASONS_ORDER]
        piv_vol = piv_vol[SEASONS_ORDER]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(piv_sent, annot=True, fmt=".1f", cmap="RdYlGn", vmin=1, vmax=5, ax=axes[0])
        style_title(axes[0], branch, "Median Sentiment (1-5)")
        
        sns.heatmap(piv_vol, annot=True, fmt=".0f", cmap="Blues", ax=axes[1])
        style_title(axes[1], branch, "Volume (Count)")
        
        plt.tight_layout()
        save_plot(f"insight1_{branch}.png", save_dir)
        plt.show()

# --- Insight 2: What Drives Low Ratings? ---
def analyze_insight_2(df, save_dir=None):
    print("\n=== Insight 2: What Drives Low Ratings? ===")
    
    # Filter Low vs High
    low_ratings = df[df['Rating'] <= 2].copy()
    if low_ratings.empty:
        print("No low ratings found.")
        return
        
    # Extract themes for low ratings
    low_ratings['themes'] = low_ratings.apply(get_themes, axis=1)
    exploded = low_ratings.explode('themes')
    
    # Count topics driving low ratings
    topic_counts = exploded['themes'].value_counts().head(10)
    
    # Viz
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=topic_counts.values, y=topic_counts.index, palette="Reds_r", hue=topic_counts.values, legend=False)
    style_title(ax, "All Parks", "Top Drivers of Low Ratings (1-2 Stars)")
    plt.xlabel("Number of Negative Reviews")
    
    save_plot("insight2_low_ratings.png", save_dir)
    plt.show()

# --- Insight 3: Seasonality Beyond Ratings ---
def analyze_insight_3(df, save_dir=None):
    print("\n=== Insight 3: Seasonality Beyond Ratings ===")
    
    # Group by Year-Month
    # Need to sort chronologically. Year_Month is string "YYYY-M".
    # Create valid date for sorting
    df = df.copy()
    # Handle YYYY-M format
    try:
        df['dt'] = pd.to_datetime(df['Year_Month'], format='%Y-%m', errors='coerce')
    except:
        # Fallback manual parse if needed
        df['dt'] = pd.to_datetime(df['Year_Month'], errors='coerce')
        
    df = df.dropna(subset=['dt'])
    
    # Aggregate per month
    monthly = df.groupby('dt').agg(
        sentiment=('sentiment_score', 'mean'),
        complaint_rate=('is_complaint', 'mean')
    ).reset_index().sort_values('dt')
    
    if monthly.empty:
        print("No valid timeline data.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(data=monthly, x='dt', y='sentiment', ax=ax1, color='green', label='Avg Sentiment')
    ax1.set_ylabel('Sentiment Score (1-5)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=monthly, x='dt', y='complaint_rate', ax=ax2, color='red', linestyle='--', label='Complaint Rate')
    ax2.set_ylabel('Complaint Rate (0-1)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    style_title(ax1, "All Parks", "Sentiment vs Complaint Rate Over Time")
    
    save_plot("insight3_seasonality.png", save_dir)
    plt.show()

# --- Insight 4: Country-Specific Expectations ---
def analyze_insight_4(df, save_dir=None):
    print("\n=== Insight 4: Country-Specific Expectations ===")
    
    # Top 10 Locations
    top_locs = df['Reviewer_Location'].value_counts().head(10).index
    subset = df[df['Reviewer_Location'].isin(top_locs)].copy()
    
    # Aggregate Price Sensitivity & Rating
    # We need to turn price_sensitivity into a metric? 
    # Or just use general sentiment? 
    # Let's check sentiment by Country
    
    agg = subset.groupby('Reviewer_Location')['sentiment_score'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=agg.values, y=agg.index, palette="viridis", hue=agg.values, legend=False)
    style_title(ax, "Global", "Average Sentiment by Visitor Country")
    plt.xlabel("Avg Sentiment Score")
    
    save_plot("insight4_country_expectations.png", save_dir)
    plt.show()

# --- Insight 5: Staff Sentiment Deep Dive ---
def analyze_insight_5(df, save_dir=None):
    print("\n=== Insight 5: Staff Sentiment Deep Dive ===")
    
    # Boxplot of Sentiment Score by Staff Tag
    # Filter rows where staff_sentiment is present
    subset = df[df['staff_sentiment'].notna() & (df['staff_sentiment'] != 'None')].copy()
    
    if subset.empty:
        print("No staff sentiment data found.")
        return
        
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='staff_sentiment', y='sentiment_score', data=subset, palette="Set2", hue='staff_sentiment', legend=False)
    style_title(ax, "All Parks", "Impact of Staff Interactions on Overall Rating")
    plt.ylabel("Overall Sentiment Score")
    
    save_plot("insight5_staff_impact.png", save_dir)
    plt.show()

# --- Insight 6: Crowding Signal Validation ---
def analyze_insight_6(df, save_dir=None):
    print("\n=== Insight 6: Crowding Signal Validation ===")
    
    # Group by Crowd Level
    # Order: Empty, Moderate, Crowded, Packed
    ORDER = ['Empty', 'Moderate', 'Crowded', 'Packed']
    
    subset = df[df['crowd_level'].isin(ORDER)].copy()
    
    if subset.empty:
        print("No crowd level data found.")
        return
        
    agg = subset.groupby('crowd_level').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        complaint_rate=('is_complaint', 'mean')
    ).reindex(ORDER)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(agg.index, agg['avg_sentiment'], color='skyblue', label='Sentiment')
    ax1.set_ylabel('Avg Sentiment', color='blue')
    ax1.set_ylim(1, 5)
    
    ax2 = ax1.twinx()
    ax2.plot(agg.index, agg['complaint_rate'], color='red', marker='o', label='Complaint %')
    ax2.set_ylabel('Complaint Rate', color='red')
    
    style_title(ax1, "All Parks", "Crowd Levels vs Sentiment & Complaints")
    
    save_plot("insight6_crowding_validation.png", save_dir)
    plt.show()

# --- Insight 7: Evidence Quotes ---
def get_evidence_quotes(df, filters, n=3):
    """
    Returns relevant quotes based on filters.
    filters: dict e.g. {'crowd_level': 'Packed', 'Branch': 'Disneyland_Paris'}
    """
    subset = df.copy()
    for col, val in filters.items():
        if col in subset.columns:
            subset = subset[subset[col] == val]
            
    # Sample
    if len(subset) > n:
        sample = subset.sample(n)
    else:
        sample = subset
        
    print(f"\n--- Evidence Quotes (Filters: {filters}) ---")
    for i, row in enumerate(sample.itertuples()):
        text = row.Review_Text[:300] + "..." if len(row.Review_Text) > 300 else row.Review_Text
        print(f"{i+1}. \"{text}\" (Rating: {row.Rating})")

# --- CX Playbook ---
def generate_cx_playbook(df, save_dir=None):
    print("\n=== CX Playbook Generation ===")
    
    # 1. Setup & Explode
    work_df = df.copy()
    if 'Season' not in work_df.columns:
        print("CRITICAL: 'Season' column missing.")
        return pd.DataFrame()
        
    work_df['extracted_themes'] = work_df.apply(get_themes, axis=1)
    # explode
    exploded = work_df.explode('extracted_themes')
    exploded = exploded[exploded['extracted_themes'].notna()]
    
    # 2. Aggregation: Group by Park, Season, Theme
    # Metrics: Count (Volume), Median Sentiment (Severity), Pct Negative
    agg = exploded.groupby(['Branch', 'Season', 'extracted_themes']).agg(
        volume=('review_uid', 'count'),
        median_sentiment=('sentiment_score', 'median'),
        neg_count=('sentiment_label', lambda x: (x == 'Negative').sum())
    ).reset_index()
    
    agg['pct_negative'] = (agg['neg_count'] / agg['volume'] * 100).round(1)
    
    # 3. Filter "Issues"
    # Rule: Volume >= 30.
    subset = agg[agg['volume'] >= 30].copy()
    
    # 4. Trend Calculation
    # Baseline: Average sentiment for that (Branch, Theme) across ALL seasons
    baseline = exploded.groupby(['Branch', 'extracted_themes'])['sentiment_score'].mean().reset_index()
    baseline.rename(columns={'sentiment_score': 'baseline_score'}, inplace=True)
    
    subset = subset.merge(baseline, on=['Branch', 'extracted_themes'], how='left')
    
    def get_trend(row):
        diff = row['median_sentiment'] - row['baseline_score']
        # If sentiment is LOWER than baseline, it's WORSE.
        if diff < -0.2: return "↑ Worse"
        elif diff > 0.2: return "↓ Better"
        else: return "≈ Same"
        
    subset['trend'] = subset.apply(get_trend, axis=1)
    
    # 5. Action Mapping
    ACTION_MAP = {
        'Queue/Crowd': 'Increase staffing, capacity smoothing, promote early-entry',
        'Staff': 'Training on problem resolution, not politeness',
        'Price': 'Improve value communication or bundle offerings',
        'Food': 'Improve peak-hour availability / quality',
        'Cleanliness': 'Increase cleaning frequency during peak',
        'Rides': 'Proactive maintenance notification & queue management',
        'Weather': 'Add more shaded/indoor rest areas',
        'Family': 'Enhance kid-friendly queuing entertainment'
    }
    subset['recommended_action'] = subset['extracted_themes'].map(ACTION_MAP).fillna("Investigate specific root cause")
    
    # 6. Evidence Quote Selection
    # Filter exploded for negative reviews
    neg_reviews = exploded[exploded['sentiment_label'] == 'Negative'].copy()
    
    def get_quote(row):
        # find matching reviews
        mask = (
            (neg_reviews['Branch'] == row['Branch']) & 
            (neg_reviews['Season'] == row['Season']) & 
            (neg_reviews['extracted_themes'] == row['extracted_themes'])
        )
        matches = neg_reviews[mask]
        
        if matches.empty:
            return "No specific negative quote found."
            
        # Pick concise one (40-300 chars)
        valid_quotes = matches[matches['Review_Text'].str.len().between(40, 300)]
        if not valid_quotes.empty:
            return valid_quotes.iloc[0]['Review_Text']
        
        # Fallback
        return matches.iloc[0]['Review_Text'][:150] + "..."

    # Prioritize: Low Sentiment (Severity) ASC, High Volume DESC
    subset.sort_values(by=['median_sentiment', 'volume'], ascending=[True, False], inplace=True)
    
    # Take top 15 "Critical Issues"
    final_df = subset.head(15).copy()
    
    # Get quotes for these 15
    final_df['representative_quote'] = final_df.apply(get_quote, axis=1)
    
    # 7. Formatting
    final_df['issue'] = final_df.apply(lambda r: f"High {r['extracted_themes']} friction", axis=1)
    final_df['evidence_metric'] = final_df['pct_negative'].astype(str) + "% negative"
    
    # Park Name Cleanup: remove "Disneyland_"
    final_df['Branch'] = final_df['Branch'].str.replace('Disneyland_', '').str.replace('_', ' ')

    output_cols = [
        'Branch',               # park (renamed later)
        'issue', 
        'median_sentiment',     # severity
        'trend', 
        'evidence_metric',      # Moved here
        'Season',               # season
        'recommended_action',
        'volume',               # vol
        'representative_quote'
    ]
    
    display_df = final_df[output_cols].rename(columns={
        'Branch': 'park',
        'extracted_themes': 'theme',
        'Season': 'season',
        'median_sentiment': 'severity',
        'volume': 'vol'
    })
    
    return display_df
