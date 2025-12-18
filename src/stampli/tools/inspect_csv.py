import pandas as pd

csv_path = "src/stampli/DisneylandReviews.csv"
try:
    # frequent encoding for this dataset
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', nrows=5)
    print("--- Columns ---")
    for col in df.columns:
        print(f"- {col}")
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Sample Row ---")
    print(df.iloc[0].to_dict())
except Exception as e:
    print(f"Error reading CSV: {e}")
