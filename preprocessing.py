import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Loads the dataset from the specified path."""
    df = pd.read_csv(path)
    return df

def print_data_structure(df):
    """Prints the structure of the DataFrame and returns it for chaining."""
    print("\n> DATA STRUCTURE & PREVIEW:")
    print("-"*40)
    print(f"Shape: {df.shape}")
    print("\n> Columns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    return df


def summary_statistics(df):
    """Prints and returns summary statistics for numeric columns."""
    df_numeric = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        'mean':     df_numeric.mean(),
        'median':   df_numeric.median(),
        'q25':      df_numeric.quantile(0.25),
        'q75':      df_numeric.quantile(0.75),
        'variance': df_numeric.var(),
        'std_dev':  df_numeric.std(),
        'mode':     df_numeric.mode().iloc[0],  # first mode if multiple
        'missing':  df_numeric.isna().sum(),
        'unique':   df_numeric.nunique(),
    })
    return stats

def explore_genres(df):
    unique_genres = df['track_genre'].unique()
    return len(unique_genres)

def check_duplicates(df):
    """Checks for duplicate track_ids and removes them to ensure data integrity."""
     # We check specifically for 'track_id' as it should be the unique identifier
    duplicate_count = df.duplicated(subset=['track_id']).sum()
    
    print("\n-----------------------------------")
    if duplicate_count > 0:
        print(f"\nWarning: Found {duplicate_count} duplicate track IDs.")
        # Keeping the first occurrence and removing the rest

        most_frequent_id = df['track_id'].value_counts().index[0]
        num_occurrences = df['track_id'].value_counts().iloc[0]
        print(f"\nInvestigating the most repeated track_id: '{most_frequent_id}' (appears {num_occurrences} times).")
        
        # Filter the dataframe to look ONLY at this specific track
        example_df = df[df['track_id'] == most_frequent_id]
        
        # Find and print the columns that have differing values across these rows
        print("Columns with DIFFERENT values for this specific track:")
        differing_cols = False

        for col in example_df.columns:
            # If a column has more than 1 unique value, it means the data changes!
            if example_df[col].nunique(dropna=False) > 1:
                differing_cols = True
                unique_vals = example_df[col].unique()
                print(f"  -> {col}: {unique_vals}")

        if not differing_cols:
            print("  -> (No differing columns found. The rows are 100% exact clones).")

        df = df.drop_duplicates(subset=['track_id'], keep='first')
        print(f"Duplicates removed. Remaining rows: {len(df)}")
    else:
        print(f"\nSuccess: No duplicate track IDs found.")
    
    print("\n-----------------------------------")
    return df

def manage_missing_values(df):
    """Handles missing values, dropping rows only if critical numeric data is missing."""
    
    # We define 'safe_columns' that can have missing values without causing critical issues. 
    # These are mostly identifiers and text fields that won't break our numeric analyses if missing.
    safe_columns = ['artists', 'album_name', 'track_name']

    df = df.replace(['-999', 'missing', 'nan', '?'], np.nan)

    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")

    initial_rows = len(df)

    critical_columns = [col for col in df.columns if col not in safe_columns]

    # Rows that will be deleted
    deleted_rows = df[df[critical_columns].isna().any(axis=1)]
    df_cleaned = df.dropna(subset=critical_columns)

    deleted_count = initial_rows - len(df_cleaned)
    print("\nArtists, album_name, and track_name can have missing values without causing critical issues.\nThese are text fields that won't break our numeric analyses if missing because we won't be using them as features. \nTherefore, we only drop rows if critical numeric data is missing.")
    print(f"\nDeleted rows (due to missing critical data): {deleted_count}")

    if deleted_count > 0:
        print("Deleted rows (showing only missing fields):")

        for idx, row in deleted_rows.iterrows():
            missing_cols = [col for col in critical_columns if pd.isna(row[col])]
            missing_info = {col: None for col in missing_cols}

            print(f"track_id: {row['track_id']}")
            print(f"missing critical data: {missing_info}")
            print("-" * 40)
    else:
        print("No rows were deleted because critical data is perfectly intact.")

    print("\n-----------------------------------")

    return df_cleaned

def prepare_columns(df):
    """Removes unnecessary columns, handles explicit casting, and trims outliers."""
    
    print(f"\n-> Dropping columns of type 'object' that won't be used as predictive features (track_id, artists, album_name, track_name).\nThese identifiers contain high-cardinality text data that standard numerical algorithms cannot process.")
    print(f"\n-> Dropping 'time_signature' feature due to near-zero variance.\nWith the 25th, 50th (median), and 75th percentiles all equal to 4, the vast majority of tracks share the same value, offering negligible predictive power for our model.")
    
    drop_columns = ['track_id', 'artists', 'album_name', 'track_name', 'time_signature']
    df_reduced = df.drop(columns=drop_columns, errors='ignore')

    print(f"\n-> Converting 'explicit' column to integer (0 or 1).")
    if 'explicit' in df_reduced.columns:
        df_reduced['explicit'] = df_reduced['explicit'].astype(int)

    print(f"\n-> Removing tracks with duration longer than 10 minutes (600,000 ms)")
    if 'duration_ms' in df_reduced.columns:
        df_reduced = df_reduced[df_reduced['duration_ms'] < 600000]
    
    # Remove tracks with 0 popularity (likely unstreamed noise)
    zero_pop_count = (df_reduced['popularity'] == 0).sum()
    print(f"\n-> Removing tracks with 0 popularity (likely unstreamed noise): {zero_pop_count} rows found.")
    df_reduced['popularity'] = df_reduced['popularity'].replace(0, np.nan)
    df_reduced = df_reduced.dropna(subset=['popularity'])

    return df_reduced

def encode_categorical(df):
    """Maps numerical keys to music notes and applies One-Hot Encoding."""
    df = df.copy()
    if 'key' in df.columns:
        key_mapping = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B', -1: 'Unknown'
        }
        df['key'] = df['key'].map(key_mapping)
        df = pd.get_dummies(df, columns=['key'], drop_first=True)

    if 'track_genre' in df.columns:
        df = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

    return df

def prepare_model_data(df_numeric):
    """Splits data into train/test sets and applies StandardScaler."""
    X = df_numeric.drop(columns=['popularity'])
    y = df_numeric['popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test