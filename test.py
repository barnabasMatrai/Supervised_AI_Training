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
    df_cleaned = df.dropna()

    print(f"Deleted rows: {initial_rows - len(df_cleaned)}")
    print("-----------------------------------")

    return df_cleaned

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
# 5. VISUALIZATION (EDA)
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
# 10. EXTRA ANALYSIS
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


# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    df = load_data(os.getcwd() + "/dataset.csv")

    print("\nPreparing columns...")
    df = prepare_columns(df)

    print("\nChecking for missing values...")
    df = manage_missing_values(df)

    summary_statistics(df)  
    
    explore_genres(df)

    df_numeric = plot_correlation(df)
    plot_boxplots(df_numeric)

    X, y, X_train, X_test, y_train, y_test = prepare_model_data(df_numeric)

    lr_model, r2_lr, mse_lr = train_linear_regression(X_train, X_test, y_train, y_test)
    tree_model, r2_tree, mse_tree = train_decision_tree(X_train, X_test, y_train, y_test)

    compare_models(r2_lr, mse_lr, r2_tree, mse_tree)

    plot_feature_importance(tree_model, X)

    loudness_energy_regression(df)

    print("\n=== SCRIPT FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
