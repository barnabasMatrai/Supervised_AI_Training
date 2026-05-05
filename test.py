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
    drop_columns = ['track_id', 'duration_ms', 'artists', 'album_name', 'track_name']
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


# ==========================================
# 3. BASIC EXPLORATION
# ==========================================
def explore_genres(df):
    print("\nUnique music genres:")
    unique_genres = df['track_genre'].unique()
    print(unique_genres)
    print(f"\nTotal unique music genres: {len(unique_genres)}")


# ==========================================
# 4. VISUALIZATION (EDA)
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


# ==========================================
# 5. DATA PREPARATION
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
# 6. MODELS
# ==========================================
def train_linear_regression(X_train, X_test, y_train, y_test):
    print("\n--- TRAINING MODEL: LINEAR REGRESSION ---")

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    return model, r2, mse


def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n--- TRAINING MODEL: DECISION TREE ---")

    base_tree = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [10, 50, 100]
    }

    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    predictions = best_tree.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    return best_tree, r2, mse


# ==========================================
# 7. COMPARISON
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
# 8. FEATURE IMPORTANCE
# ==========================================
def plot_feature_importance(model, X):
    print("\nGenerating Feature Importance Chart...")

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. EXTRA ANALYSIS
# ==========================================
def loudness_energy_regression(df):
    print("\n--- EXTRA: Loudness vs Energy ---")

    X = df['loudness'].values.reshape(-1, 1)
    y = df['energy'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 1)

    mse = np.mean((y - y_pred) ** 2)
    print("Mean Squared Error:", mse)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='True values')
    plt.scatter(X, y_pred, marker='x', label='Predictions')

    plt.xlabel('Loudness')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Linear Regression: Loudness vs Energy')
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
# MAIN PIPELINE
# ==========================================
def main():
    df = load_data(os.getcwd() + "\\dataset.csv")

    print("\nChecking for missing values before preparing the columns...")
    df = manage_missing_values(df)

    print("\nPreparing columns...")
    df = prepare_columns(df)

    print("\nChecking for missing values after cleaning...")
    df = manage_missing_values(df)

    print(df['popularity'].value_counts().head(10))
    df['popularity'] = df['popularity'].replace(0, np.nan)
    df = df.dropna(subset=['popularity'])
    plot_histogram(df, "popularity")

    """explore_genres(df)

    df_numeric = plot_correlation(df)
    plot_boxplots(df_numeric)

    X, y, X_train, X_test, y_train, y_test = prepare_model_data(df_numeric)

    lr_model, r2_lr, mse_lr = train_linear_regression(X_train, X_test, y_train, y_test)
    tree_model, r2_tree, mse_tree = train_decision_tree(X_train, X_test, y_train, y_test)

    compare_models(r2_lr, mse_lr, r2_tree, mse_tree)

    plot_feature_importance(tree_model, X)

    loudness_energy_regression(df)

    print("\n=== SCRIPT FINISHED SUCCESSFULLY ===")"""


if __name__ == "__main__":
    main()
