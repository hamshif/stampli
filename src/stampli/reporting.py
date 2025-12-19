
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os
from datetime import datetime
from stampli.disney_viz import get_playbook_narrative_from_df

# ... (Previous explanations code omitted for brevity)

# ... (generate_excel_playbook function omitted)


def generate_pdf_report(plot_dir, output_path):
    """
    Creates a PDF report combining text explanations and the saved PNG plots.
    Includes an automated statistical narrative summary from the playbook.
    """
    print(f"Generating PDF Report: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- Auto Narrative Generation ---
    narrative_text = ""
    playbook_excel_path = os.path.join(plot_dir, "disney_cx_playbook.xlsx")
    
    if os.path.exists(playbook_excel_path):
        try:
            # Read back the playbook we just generated
            df_playbook = pd.read_excel(playbook_excel_path)
            # Use shared logic
            narrative_text = get_playbook_narrative_from_df(df_playbook)
                
        except Exception as e:
            narrative_text += f"Could not load playbook stats: {e}"
            print(f"Error reading playbook for stats: {e}")
    else:
        narrative_text += "Playbook file not found for statistical summary."
        print(f"Playbook Excel not found at: {playbook_excel_path}")

    # --- PDF Creation ---
    with PdfPages(output_path) as pdf:
        # --- Title Page ---
        fig = plt.figure(figsize=(11, 8.5)) # Letter sizeish
        fig.text(0.5, 0.6, "Stampli CX Insights Report", ha='center', fontsize=30, fontweight='bold')
        fig.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
        # --- Narrative Summary Page ---
        fig_narrative = plt.figure(figsize=(11, 8.5))
        fig_narrative.text(0.1, 0.85, "Executive Narrative Summary", fontsize=20, fontweight='bold')
        fig_narrative.text(0.1, 0.4, narrative_text, fontsize=12, wrap=True, fontfamily='monospace') # monospace for list alignment
        plt.axis('off')
        pdf.savefig(fig_narrative)
        plt.close(fig_narrative)
        
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
                    print(f"Warning: Plot not found {os.path.abspath(fpath)}")
                    
    print("PDF Report generated successfully.")
