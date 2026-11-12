import pandas as pd
import string

df = pd.read_csv("Vision2Braille/data/gestures.csv")
valid_letters = [c for c in string.ascii_uppercase if c != 'Q']

print("Unique labels found:", df['label'].unique())
print()

bad_rows = df[~df['label'].isin(valid_letters)]
print("Bad rows found:", len(bad_rows))
print(bad_rows[['label']] if len(bad_rows) > 0 else "None")
