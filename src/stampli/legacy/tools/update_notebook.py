import json
from pathlib import Path

NB_PATH = "/home/gideon/dev/biocontext/biocontext-services/src/stampli/disney_exploration.ipynb"

def append_cell():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r') as f:
        nb = json.load(f)

    new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Specific Investigation: California in June\n",
                "# User Question: \"Is Disneyland California usually crowded in June?\"\n",
                "\n",
                "# 1. Parse Month\n",
                "def get_month(ym):\n",
                "    try:\n",
                "        return int(ym.split('-')[1])\n",
                "    except:\n",
                "        return 0\n",
                "\n",
                "pd_disney['month'] = pd_disney['Year_Month'].apply(get_month)\n",
                "\n",
                "# 2. Filter CA + June\n",
                "ca_june = pd_disney[\n",
                "    (pd_disney['Branch'] == 'Disneyland_California') & \n",
                "    (pd_disney['month'] == 6)\n",
                "]\n",
                "\n",
                "print(f\"Total Reviews (CA + June): {len(ca_june)}\")\n",
                "print(\"\\nExtracted Crowd Levels:\")\n",
                "print(ca_june['crowd_level'].value_counts(dropna=False))\n",
                "\n",
                "# 3. False Negative Check\n",
                "KEYWORDS = [\"crowd\", \"packed\", \"busy\", \"line\", \"queue\", \"wait\"]\n",
                "mask_keys = ca_june['Review_Text'].str.contains('|'.join(KEYWORDS), case=False, na=False)\n",
                "missed_ca_june = ca_june[mask_keys & ca_june['crowd_level'].isna()]\n",
                "\n",
                "print(f\"\\nPotential False Negatives (Keywords present, Label Missing): {len(missed_ca_june)}\")\n",
                "if not missed_ca_june.empty:\n",
                "    display_scrollable_dataframe(missed_ca_june[['Review_Text', 'summary']])"
            ]
        }
    
    nb['cells'].append(new_cell)
    
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=4)

    print("Appended cell successfully.")

if __name__ == "__main__":
    append_cell()
