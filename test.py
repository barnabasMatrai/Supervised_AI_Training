
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data(path):
    df = pd.read_csv(path)
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.head())
    return df


# ==========================================
# 2. CLEANING
# ==========================================
def prepare_columns(df):
    drop_columns = ['track_id', 'duration_ms', 'artists', 'album_name', 'track_name', 'time_signature']
    df_reduced = df.drop(columns=drop_columns)

    if 'explicit' in df_reduced.columns:
        df_reduced['explicit'] = df_reduced['explicit'].astype(int)

    return df_reduced


def manage_missing_values(df):
    df = df.replace(['-999', 'missing', 'nan', '?'], np.nan)

    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        print("Missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    initial_rows = len(df)

    # Rows that will be deleted
    deleted_rows = df[df.isna().any(axis=1)]
    df_cleaned = df.dropna()

    deleted_count = initial_rows - len(df_cleaned)
    print(f"Deleted rows: {deleted_count}")

    if deleted_count > 0:
        print("\nDeleted rows (showing only missing fields):")

        for idx, row in deleted_rows.iterrows():
            missing_cols = row[row.isna()].index.tolist()
            missing_info = {col: None for col in missing_cols}

            print(f"track_id: {row['track_id']}")
            print(f"missing: {missing_info}")
            print("-" * 40)
    else:
        print("No rows were deleted.")

    print("-----------------------------------")

    return df_cleaned

def check_duplicates(df):
    """Checks for duplicate track_ids and removes them to ensure data integrity."""
    # We check specifically for 'track_id' as it should be the unique identifier
    duplicate_count = df.duplicated(subset=['track_id']).sum()
    
    if duplicate_count > 0:
        print(f"Warning: Found {duplicate_count} duplicate track IDs.")
        # Keeping the first occurrence and removing the rest

        top_duplicates = df['track_id'].value_counts().head(3)
        print(f"Most frequent track IDs:\n{top_duplicates}")

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
        print("Success: No duplicate track IDs found.")
    
    print("-----------------------------------")
    return df

# ==========================================
# 2b. ENCODING
# ==========================================
def encode_categorical(df):
    df = df.copy()

    if 'track_genre' in df.columns:
        df = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

    return df

# ==========================================
# 3. SUMMARY STATISTICS
# ==========================================
def summary_statistics(df):
    print("\n--- SUMMARY STATISTICS (Numeric Columns) ---")
    
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
    
    print(stats.to_string())
    print("\n")
    return stats

# ==========================================
# 4. BASIC EXPLORATION
# ==========================================
def explore_genres(df):
    print("\nUnique music genres:")
    unique_genres = df['track_genre'].unique()
    print(unique_genres)
    print(f"\nTotal unique music genres: {len(unique_genres)}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_correlation(df):
    print("Generating Correlation Map...")
    df_numeric = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 10))
    correlation = df_numeric.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return df_numeric


def plot_boxplots(df_numeric):
    print("Generating Boxplot...")
    plt.figure(figsize=(15, 8))
    df_numeric.boxplot()
    plt.xticks(rotation=45)
    plt.title("Distribution of Numeric Features")
    plt.tight_layout()
    plt.show()

def plot_histograms(df_numeric):
    print("Generating Histograms...")
    df_numeric.hist(figsize=(15, 10), bins=30, edgecolor='black')
    plt.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_density(df_numeric):
    print("Generating Density Plots...")
    cols = ['popularity', 'energy', 'loudness', 'danceability', 'valence', 'tempo']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        df_numeric[col].plot(kind='kde', ax=axes[i], color='steelblue')
        axes[i].set_title(f'Density: {col}')
        axes[i].set_xlabel(col)
    
    plt.suptitle("Density Plots", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_scatter(df):
    print("Generating Scatter Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Energy vs Loudness
    axes[0].scatter(df['loudness'], df['energy'], alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Loudness')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Loudness vs Energy')
    
    # Danceability vs Popularity
    axes[1].scatter(df['danceability'], df['popularity'], alpha=0.3, color='coral')
    axes[1].set_xlabel('Danceability')
    axes[1].set_ylabel('Popularity')
    axes[1].set_title('Danceability vs Popularity')
    
    # Tempo vs Energy
    axes[2].scatter(df['tempo'], df['energy'], alpha=0.3, color='mediumseagreen')
    axes[2].set_xlabel('Tempo')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Tempo vs Energy')
    
    plt.suptitle("Scatter Plots", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_histogram(df, column):
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=30)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. DATA PREPARATION
# ==========================================
def prepare_model_data(df_numeric):
    print("\n--- DATA PREPARATION ---")

    X = df_numeric.drop(columns=['popularity'])
    y = df_numeric['popularity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Scaling complete.")

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test


# ==========================================
# 7. MODELS
# ==========================================

def linear_regression(df, feature_col, target_col):
    print(f"\n--- SIMPLE LINEAR REGRESSION: {feature_col} vs {target_col} ---")

    X = df[feature_col].values.reshape(-1, 1)
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train  )

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_test, y_test, alpha=0.3, color='blue', label='True Values (Test Data)')
    
    X_line = np.array([[X_test.min()], [X_test.max()]])
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=3, label='Regression Line')

    plt.xlabel(feature_col)
    plt.ylabel(target_col
    )
    plt.legend()
    plt.title(f'Linear Regression: {feature_col} vs {target_col}')
    plt.show()
    return model, r2, mse

def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n--- TRAINING MODEL: DECISION TREE ---")

    base_tree = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [10, 50, 100],
        'min_samples_leaf': [10, 20, 50]
    }

    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    predictions = best_tree.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"Decision Tree R-squared (R2): {r2:.4f}")
    print(f"Decision Tree Mean Squared Error (MSE): {mse:.4f}")
    return best_tree, r2, mse

# ==========================================
# 7. CROSS VALIDATION
# ==========================================
def evaluate_model_cv(model, X, y, name):
    print(f"\n--- CROSS VALIDATION: {name} ---")

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"Scores: {scores}")
    print(f"Mean R2: {scores.mean():.4f}")

    return scores.mean()

# ==========================================
# 8. COMPARISON
# ==========================================
def compare_models(r2_lr, mse_lr, r2_tree, mse_tree):
    print("\n==========================================")
    print("🏆 FINAL SHOWDOWN: PREDICTING POPULARITY 🏆")
    print("==========================================")

    print("1. LINEAR REGRESSION")
    print(f"   - R2: {r2_lr:.4f}")
    print(f"   - MSE: {mse_lr:.2f}")

    print("\n2. DECISION TREE")
    print(f"   - R2: {r2_tree:.4f}")
    print(f"   - MSE: {mse_tree:.2f}")

    improvement = (r2_tree / r2_lr) if r2_lr > 0 else 0
    print(f"\nDecision Tree is {improvement:.1f}x better (R2 ratio).")


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================
def plot_feature_importance(model, X):
    print("\nGenerating Feature Importance Chart...")

    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    df = load_data(os.getcwd() + "/dataset.csv")

    print("\n Checking for duplicate Track IDs...")
    df = check_duplicates(df)

    print("\nChecking for missing values before preparing the columns...")
    df = manage_missing_values(df)

    explore_genres(df)

    print("\nPreparing columns...")
    df = prepare_columns(df)
    
    print("\nChecking for missing values after cleaning...")
    df = manage_missing_values(df)

    summary_statistics(df)  
    
    print(df['popularity'].value_counts().head(10))
    df['popularity'] = df['popularity'].replace(0, np.nan)
    df = df.dropna(subset=['popularity'])

    df_numeric = df.select_dtypes(include=[np.number])
    plot_correlation(df_numeric)
    plot_boxplots(df_numeric)
    plot_histograms(df_numeric)
    plot_histogram(df, "popularity")
    plot_density(df_numeric)
    plot_scatter(df)

    _, r2_lr_loudness, mse_lr_loudness = linear_regression(df_numeric, 'loudness', 'energy')
    _, r2_lr, mse_lr = linear_regression(df_numeric, 'loudness', 'popularity')

    df_encoded = encode_categorical(df)
    df_tree = df_encoded.select_dtypes(include=[np.number])

    X, y, X_train, X_test, y_train, y_test = prepare_model_data(df_tree)

    evaluate_model_cv(LinearRegression(), X, y, "Linear Regression")

    tree_model, r2_tree, mse_tree = train_decision_tree(X_train, X_test, y_train, y_test)

    compare_models(r2_lr, mse_lr, r2_tree, mse_tree)

    plot_feature_importance(tree_model, X)

    print("\n=== SCRIPT FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
