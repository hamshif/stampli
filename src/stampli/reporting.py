
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os
from datetime import datetime

# --- Explanations for the Report ---
INSIGHT_EXPLANATIONS = {
    "Insight 1": (
        "Theme Sentiment Analysis",
        "Heatmaps regarding median sentiment and volume across seasons and themes.\n"
        "Key Takeaway: Certain themes (like Crowding) consistently drive lower sentiment,\n"
        "while seasonality affects volume significantly."
    ),
    "Insight 2": (
        "Drivers of Low Ratings",
        "Analysis of reviews with 1-2 stars to identify primary complaints.\n"
        "Key Takeaway: The 'Queue/Crowd' and 'Price' themes are the most frequent\n"
        "drivers of negative experiences."
    ),
    "Insight 3": (
        "Seasonality Trends",
        "Tracking sentiment and complaint rates over time.\n"
        "Key Takeaway: Sentiment dips often correlate with peak seasons and high complaint rates,\n"
        "suggesting operational strain during busy periods."
    ),
    "Insight 4": (
        "Country-Specific Expectations",
        "Average sentiment broken down by visitor country.\n"
        "Key Takeaway: Visitors from different regions have varying baseline satisfaction levels,\n"
        "which should inform targeted expectation management."
    ),
    "Insight 5": (
        "Staff Impact",
        "Impact of staff interactions on overall sentiment.\n"
        "Key Takeaway: Positive staff interactions ('Helpful', 'Friendly') significantly boost\n"
        "overall ratings, while negative ones ('Rude') are detrimental."
    ),
    "Insight 6": (
        "Crowding Validation",
        "Correlation between reported crowd levels and sentiment.\n"
        "Key Takeaway: 'Packed' conditions correlate strongly with lower sentiment scores,\n"
        "validating the impact of overcrowding on guest experience."
    )
}

def generate_excel_playbook(df, output_path):
    """
    Saves the dataframe to Excel with frozen top row and first column.
    """
    print(f"Generating Excel Playbook: {output_path}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # standard pandas write
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='CX Playbook')
            
            # Access workbook/worksheet to freeze panes
            workbook = writer.book
            worksheet = writer.sheets['CX Playbook']
            
            # Freeze Top Row and First 6 Columns (G2 means freeze above and left of G2)
            worksheet.freeze_panes = 'G2'
            
            # Auto-adjust column widths (simple estimation)
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                
                # Special handling for 'recommended_action': make it 1.3x wider
                # Column header is in the first cell
                if str(column[0].value) == 'recommended_action':
                    adjusted_width = adjusted_width * 1.3
                
                # Cap width (except maybe for recommended action if we want it really wide? 
                # Let's cap at 65 for it (50 * 1.3) roughly, or just respect general cap but scaling it before?)
                # User's request 'make it 1.3 times wider' implies relative to what it would be.
                # If it's hitting the cap of 50, it stays 50. 
                # Let's apply cap *after* multiplication for this column if needed, or simply increase cap.
                
                limit = 50
                if str(column[0].value) == 'recommended_action':
                    limit = 80 # Allow it to be wider
                
                if adjusted_width > limit: adjusted_width = limit
                
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                
        print("Excel Playbook saved successfully.")
        
    except Exception as e:
        print(f"Failed to save Excel Playbook: {e}")


def generate_pdf_report(plot_dir, output_path):
    """
    Creates a PDF report combining text explanations and the saved PNG plots.
    """
    print(f"Generating PDF Report: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # --- Title Page ---
        fig = plt.figure(figsize=(11, 8.5)) # Letter sizeish
        fig.text(0.5, 0.6, "Stampli CX Insights Report", ha='center', fontsize=30, fontweight='bold')
        fig.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
        # --- Insight Pages ---
        # Map insight keys to expected filenames prefixes
        # This relies on the filenames generated in disney_viz.py
        
        insight_map = [
            ("Insight 1", ["insight1_Disneyland_California.png", "insight1_Disneyland_Paris.png", "insight1_Disneyland_HongKong.png"]),
            ("Insight 2", ["insight2_low_ratings.png"]),
            ("Insight 3", ["insight3_seasonality.png"]),
            ("Insight 4", ["insight4_country_expectations.png"]),
            ("Insight 5", ["insight5_staff_impact.png"]),
            ("Insight 6", ["insight6_crowding_validation.png"]),
        ]
        
        for insight_title, files in insight_map:
            if insight_title not in INSIGHT_EXPLANATIONS: continue
            
            header, explanation = INSIGHT_EXPLANATIONS[insight_title]
            
            # 1. Text Page for the Insight
            # Using figure to place text
            fig_text = plt.figure(figsize=(11, 8.5))
            fig_text.text(0.1, 0.8, f"{insight_title}: {header}", fontsize=20, fontweight='bold')
            fig_text.text(0.1, 0.7, explanation, fontsize=14, wrap=True)
            plt.axis('off')
            
            # If there's only 1 image, maybe put it on the same page?
            # Let's try to fit 1 image on the text page if possible, otherwise separate.
            # For simplicity and robustness with multiple images (Insight 1 has 3), 
            # let's do Text Page -> Image Page(s).
            
            pdf.savefig(fig_text)
            plt.close(fig_text)
            
            # 2. Image Pages
            for fname in files:
                fpath = os.path.join(plot_dir, fname)
                if os.path.exists(fpath):
                    try:
                        img = mpimg.imread(fpath)
                        
                        # Calculate aspect ratio
                        h, w, c = img.shape
                        fig_img = plt.figure(figsize=(11, 8.5))
                        
                        # Display image
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(fname, fontsize=10) # Simple filename title
                        
                        pdf.savefig(fig_img)
                        plt.close(fig_img)
                    except Exception as e:
                        print(f"Error embedding {fname}: {e}")
                else:
                    print(f"Warning: Plot not found {fpath}")
                    
    print("PDF Report generated successfully.")
