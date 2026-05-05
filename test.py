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
# 1. CLEANING FUNCTIONS
# ==========================================
def prepare_columns(df):
    drop_columns = ['track_id', 'duration_ms','artists', 'album_name', 'track_name']
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

# ==========================================
# 2. ADVANCED EDA
# ==========================================

print("Generating Correlation Map")
# We select only numerical data (essential for correlation)
df_numeric = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
correlation = df_numeric.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("Generating Boxplot... (Close the window to continue)")
plt.figure(figsize=(15, 8))
df_numeric.boxplot()
plt.xticks(rotation=45)
plt.title("Distribution numeric features")
plt.tight_layout()
plt.show()

# ==========================================
# 3. DATA PREPARATION & SCALING
# ==========================================
print("\n--- DATA PREPARATION ---")
print("Target: Popularity. Preparing data for model comparison...")

X = df_numeric.drop(columns=['popularity'])
y = df_numeric['popularity']

# splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling the features (important for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.")

# ==========================================
# 4. MODEL 1: LINEAR REGRESSION (BASELINE)
# ==========================================
print("\n--- TRAINING MODEL 1: LINEAR REGRESSION ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)
r2_lr = r2_score(y_test, lr_predictions)
mse_lr = mean_squared_error(y_test, lr_predictions)

# ==========================================
# 5. MODEL 2: DECISION TREE (CHALLENGER)
# ==========================================
print("--- TRAINING MODEL 2: DECISION TREE ---")

# Defining base model and grid of parameters to test
base_tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 50, 100]
}

# Cross-Validation search
grid_search = GridSearchCV(estimator=base_tree, 
                           param_grid=param_grid, 
                           cv=3, 
                           scoring='neg_mean_squared_error',
                           verbose=1)

grid_search.fit(X_train_scaled, y_train)

# Final model with best parameters
best_tree = grid_search.best_estimator_
print(f"\nBest Hyperparameters identified: {grid_search.best_params_}")

# Final Performance Assessment
tree_pred = best_tree.predict(X_test_scaled)
r2_tree = r2_score(y_test, tree_pred)
mse_tree = mean_squared_error(y_test, tree_pred)

# ==========================================
# 6. COMPARISON & RESULTS
# ==========================================
print("\n==========================================")
print("🏆 FINAL SHOWDOWN: PREDICTING POPULARITY 🏆")
print("==========================================")
print(f"1. LINEAR REGRESSION")
print(f"   - R-squared (R2): {r2_lr:.4f}")
print(f"   - Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"\n2. DECISION TREE (Tuned)")
print(f"   - R-squared (R2): {r2_tree:.4f}")
print(f"   - Mean Squared Error (MSE): {mse_tree:.2f}")
print("==========================================")

# calculating the improvement factor of the Decision Tree over Linear Regression
improvement = (r2_tree / r2_lr) if r2_lr > 0 else 0
print(f"\nConclusion: The Decision Tree is approximately {improvement:.1f}x better at capturing the complex, non-linear rules of music popularity compared to the Linear model!")

# Feature importance chart
print("\nGenerating Feature Importance Chart from the winning model... (Close the window to finish)")
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_tree.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Decision Tree Feature Importance: What drives Popularity?")
plt.tight_layout()
plt.show()

print("\n=== SCRIPT FINISHED SUCCESSFULLY ===")

# df.boxplot()
# plt.show()

X = df['loudness'].values.reshape(-1, 1)
y = df['energy'].values

model = LinearRegression()
model.fit(X, y)

# predictions
y_pred = model.predict(X)
y_pred = np.clip(y_pred, 0, 1)

# mean squared error
mse = np.mean((y - y_pred) ** 2)
print("Mean Squared Error:", mse)

# visualise the residuals

plt.figure(figsize=(8, 6))

# true values (blue circles)
plt.scatter(X, y, color='blue', label='True values')

# predicted values (red x)
plt.scatter(X, y_pred, color='red', marker='x', label='Predictions')

# residual lines (green)
# for i in range(len(X)):
#     plt.plot([X[i], X[i]], [y[i], y_pred[i]], color='green')

plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.legend()
plt.title('Linear Regression with Residuals')

plt.show()