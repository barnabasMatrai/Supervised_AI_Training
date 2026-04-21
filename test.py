import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def prepare_columns(df):
    drop_columns = ['track_id', 'duration_ms',] #artists, album_name, track_name
    df_reduced = df.drop(columns=drop_columns)

    # converting boolean variables (True/False -> 1/0)
    if 'explicit' in df_reduced.columns:
        df_reduced['explicit'] = df_reduced['explicit'].astype(int)
        
    return df_reduced

def manage_missing_values(df):
   
    df = df.replace(['-999', 'missing', 'nan', '?'], np.nan)
    
    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        print("missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("no missing values found.")
        
    initial_rows = len(df)
    df_cleaned = df.dropna()
    
    print(f"Deleted rows: {initial_rows - len(df_cleaned)}")
    print("-----------------------------------")
    
    return df_cleaned



df = pd.read_csv(os.getcwd() + "\\dataset.csv")

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.head())

print("Preparing columns...")
df = prepare_columns(df)
print("\nchecking for missing values...")
df = manage_missing_values(df)

print("\nUnique music genres:")
unique_genres = df['track_genre'].unique()
print(unique_genres)

print(f"\nTotal unique music genres: {len(unique_genres)}")

# class_counts = df['popularity'].value_counts()

# class_counts.plot(kind='bar')

# plt.title("Class Distribution")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()

df.boxplot()
plt.show()