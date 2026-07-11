# inspect_features.py
import pandas as pd
import sys

path = "floodsense_features_engineered.xlsx"
try:
    df = pd.read_excel(path)
    print("Columns:")
    for col in df.columns:
        print(col)
except Exception as e:
    print("Error loading file:", e)
    sys.exit(1)
