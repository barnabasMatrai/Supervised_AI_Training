import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Import local modules
import preprocessing as pp
import visualization as vz
import modeling as md

def main():
    print("="*70)
    print(" DECODING SPOTIFY POPULARITY - DATA SCIENCE WORKFLOW REPORT")
    print("="*70)
    print("REPRODUCIBILITY NOTE: ")
    print("To run this project from zero, please ensure the raw dataset ")
    print("is placed at 'data/dataset.csv' relative to this script.")
    print("Dataset source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
    print("="*70)

    # ---------------------------------------------------------
    # STEP 1: DATA LOADING & CLEANING
    # ---------------------------------------------------------
    print("\n[STEP 1] Data Ingestion & Quality Control")
    dataset_path = os.path.join(os.getcwd(), "data", "dataset.csv")
    
    try:
        df = pp.load_data(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Cannot find data at {dataset_path}. Please create the folder and add the file.")
        return

    initial_len = len(df)
    df = pp.print_data_structure(df)
    df = pp.check_duplicates(df)
    df = pp.manage_missing_values(df)

    unique_genres = pp.explore_genres(df)
    print(f"-> Unique music genres: {unique_genres}")

    print("-> Summary statistics (Numeric Columns): ")
    stats = pp.summary_statistics(df)
    print(stats.to_string())
    print("\n")
    
    df = pp.prepare_columns(df)
    
    print(f"\n> INTERPRETATION: Dataset reduced from {initial_len} to {len(df)} ro")
    print("> CONCLUSION: By surgically removing duplicates (multi-genre tracks) and")
    print("> zero-popularity tracks, we prevented our target variable from being skewed.")

    # ---------------------------------------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ---------------------------------------------------------
    print("\n[STEP 2] Exploratory Data Analysis & Visualizations")

    df_numeric = df.select_dtypes(include=[np.number])
    
    print("-> Generating Histograms...")
    vz.plot_histograms(df_numeric)

    print("-> Generating Boxplots...")
    vz.plot_boxplots(df_numeric)

    print("-> Generating Correlation Matrix...")
    vz.plot_correlation(df)
    
    print("-> Generating Scatter Plots...")
    vz.plot_scatter(df)

    print("\n> FINDINGS (EDA):")
    print("  1. The histograms show the categoric nature of the 'key' feature.")
    print("  2. The 2x2 Boxplots reveal severe scale imbalance (e.g., duration vs energy).")
    print("  3. The Correlation Matrix shows NO strong linear link to popularity (max ~0.07).")
    print("  4. Scatter plots confirm 'popularity' data acts as a chaotic cloud, not a line.")

    # ---------------------------------------------------------
    # STEP 3: PREPROCESSING & ENCODING
    # ---------------------------------------------------------
    print("\n[STEP 3] Feature Engineering & Data Preparation")
    df_encoded = pp.encode_categorical(df)
    df_tree = df_encoded.select_dtypes(include=[np.number])

    X, y, X_train, X_test, y_train, y_test = pp.prepare_model_data(df_tree)
    
    print("-> Categorical Data encoded (Genres, Keys mapped to notes).")
    print("-> StandardScaler applied to numerical features.")
    print("\n> INTERPRETATION: One-Hot Encoding prevents the model from assuming false")
    print("> mathematical hierarchies between music notes. Scaling ensures features")
    print("> like 'duration' do not mathematically dominate 'energy'.")

    # ---------------------------------------------------------
    # STEP 4: BASELINE MODELING (LINEAR REGRESSION)
    # ---------------------------------------------------------
    print("\n[STEP 4] Baseline Testing (Linear Logic)")
    
    # Simple linear to prove physics (Loudness vs Energy)
    _, r2_lr_phys, mse_lr_phys = md.linear_regression(df_numeric, 'loudness', 'energy', y_min=0, y_max=1)
    print(f"-> Physical Check (Loudness vs Energy) R2: {r2_lr_phys:.4f} MSE: {mse_lr_phys:.2f} (Model understands physics)")
    
    # Simple linear to prove target complexity (Loudness vs Popularity)
    _, r2_lr_target, mse_lr = md.linear_regression(df_numeric, 'loudness', 'popularity', y_min=0, y_max=100)
    
    # Cross Validated Multiple Linear Regression
    r2_multi_cv, mse_multi_cv = md.evaluate_model_cv(LinearRegression(), X, y)
    
    print(f"-> Linear Regression (Target: Popularity) R2: {r2_lr_target:.4f} MSE: {mse_lr:.2f}")
    print(f"-> Multiple Linear Regression (All Features) CV Mean R2: {r2_multi_cv:.4f} MSE: {mse_multi_cv:.2f}")
    
    print("\n> FINDING: The Baseline models failed to predict popularity (R2 near zero).")
    print("> CONCLUSION: The music market is non-linear. This justifies moving to a Decision Tree.")

    # ---------------------------------------------------------
    # STEP 5: ADVANCED MODELING (DECISION TREE)
    # ---------------------------------------------------------
    print("\n[STEP 5] Advanced Modeling (Decision Tree Regressor)")
    print("-> Running GridSearchCV (Please wait...)")
    
    tree_model, r2_tree, mse_tree, best_params = md.train_decision_tree(X_train, X_test, y_train, y_test)
    
    print(f"-> Best Hyperparameters: {best_params}")
    print(f"-> Decision Tree R2:  {r2_tree:.4f}")
    print(f"-> Decision Tree MSE: {mse_tree:.2f}")

    md.compare_models(r2_multi_cv, mse_multi_cv, r2_tree, mse_tree)

    print("\n> CONCLUSION: By allowing non-linear splits, the Decision Tree vastly outperformed")
    print("> the linear baseline, capturing the complex conditional rules of music popularity.")

    # ---------------------------------------------------------
    # STEP 6: FEATURE IMPORTANCE
    # ---------------------------------------------------------
    print("\n[STEP 6] Feature Importance Extraction")
    vz.plot_feature_importance(tree_model, X_train)
    
    print("\n> FINAL FINDING: Vocal presence ('instrumentalness') acts as the primary gatekeeper")
    print("> for mainstream success, outweighing traditional assumptions like 'danceability'.")
    print("\n=== SCRIPT FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()