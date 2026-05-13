import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(df, feature_col, target_col, y_min=None, y_max=None):
    """Trains a simple linear regression and plots the result."""
    X = df[feature_col].values.reshape(-1, 1)
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, alpha=0.3, color='blue', label='True Values')
    
    X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_line_raw = model.predict(X_line)

    if y_min is not None and y_max is not None:
        y_line_final = np.clip(y_line_raw, y_min, y_max)
    else:
        y_line_final = y_line_raw 

    plt.plot(X_line, y_line_final, color='red', linewidth=3, label='Regression Line')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.title(f'Linear Regression: {feature_col} vs {target_col}')
    plt.show()
    
    return model, r2, mse

def evaluate_model_cv(model, X, y, cv_folds=5):
    """Evaluates a model using K-Fold Cross Validation for both R2 and MSE."""
    
    scoring = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error' 
    }
    
    scores = cross_validate(model, X, y, cv=cv_folds, scoring=scoring)
    
    mean_r2 = scores['test_r2'].mean()
    mean_mse = -scores['test_mse'].mean()
    
    return mean_r2, mean_mse

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Trains a Decision Tree using GridSearchCV for hyperparameter tuning."""
    base_tree = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [10, 50, 100],
        'min_samples_leaf': [10, 20, 50]
    }
    grid_search = GridSearchCV(estimator=base_tree, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    predictions = best_tree.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    return best_tree, r2, mse, grid_search.best_params_

def compare_models(r2_lr, mse_lr, r2_tree, mse_tree):
    print("\n==========================================")
    print("FINAL SHOWDOWN: PREDICTING POPULARITY")
    print("==========================================")

    print("1. LINEAR REGRESSION")
    print(f"   - R2: {r2_lr:.4f}")
    print(f"   - MSE: {mse_lr:.2f}")

    print("\n2. DECISION TREE")
    print(f"   - R2: {r2_tree:.4f}")
    print(f"   - MSE: {mse_tree:.2f}")
