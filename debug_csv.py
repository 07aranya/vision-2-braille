import os
import string
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "gestures.csv")

df = pd.read_csv(CSV_PATH)
valid_letters = list(string.ascii_uppercase)

print(f"Total rows: {len(df)}")
print("Unique labels found:", sorted(df['label'].unique().tolist()))

missing = [c for c in valid_letters if c not in df['label'].values]
print("Missing letters:", missing if missing else "None (all 26 present!)")

bad_rows = df[~df['label'].isin(valid_letters)]
print("Bad rows found:", len(bad_rows))
if len(bad_rows) > 0:
    print(bad_rows[['label']])
