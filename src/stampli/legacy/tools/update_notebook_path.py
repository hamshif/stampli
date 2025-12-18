import json
from pathlib import Path

NB_PATH = "/home/gideon/dev/biocontext/biocontext-services/src/stampli/disney_exploration.ipynb"

def update_path():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r') as f:
        nb = json.load(f)

    # Simple text replacement in the source code of the cell defining ENRICHED_PATH
    # We look for the cell containing "ENRICHED_PATH ="
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "ENRICHED_PATH =" in line and "reviews_enriched_v1" in line:
                    # Replace with V4
                    new_line = line.replace("reviews_enriched_v1", "reviews_enriched_backfill_v4")
                    new_source.append(new_line)
                    found = True
                    print("Found and updated ENRICHED_PATH line.")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if found:
        with open(NB_PATH, 'w') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("Could not find ENRICHED_PATH line to update.")

if __name__ == "__main__":
    update_path()
