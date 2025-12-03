#!/usr/bin/env python3
import pandas as pd

print("Checking balanced_dataset.csv...")
df = pd.read_csv('data/standardized/balanced_dataset.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"NaN values per column:")
print(df.isnull().sum())
print(f"\nSample data:")
print(df.head())
print(f"\nUnique values in last column:")
if len(df.columns) > 0:
    last_col = df.columns[-1]
    print(f"Column '{last_col}': {df[last_col].unique()}")
    print(f"Value counts:")
    print(df[last_col].value_counts())