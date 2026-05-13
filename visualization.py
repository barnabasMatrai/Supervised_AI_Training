import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlation(df):
    """Plots a heatmap of numeric feature correlations."""
    df_numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    correlation = df_numeric.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_boxplots(df_numeric):
    """Plots boxplots grouped by feature scales to highlight scale imbalance."""
    cols_0_1 = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'speechiness']
    cols_100 = ['popularity', 'tempo']
    cols_neg = ['loudness']
    cols_huge = ['duration_ms']

    cols_0_1 = [col for col in cols_0_1 if col in df_numeric.columns]
    cols_100 = [col for col in cols_100 if col in df_numeric.columns]
    cols_neg = [col for col in cols_neg if col in df_numeric.columns]
    cols_huge = [col for col in cols_huge if col in df_numeric.columns]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    if cols_0_1:
        df_numeric[cols_0_1].boxplot(ax=axes[0])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_title("Audio Features (Scale 0 to 1)", fontsize=12)
    if cols_100:
        df_numeric[cols_100].boxplot(ax=axes[1])
        axes[1].set_title("Popularity & Tempo (Scale 0-250)", fontsize=12)
    if cols_neg:
        df_numeric[cols_neg].boxplot(ax=axes[2])
        axes[2].set_title("Loudness (Negative Scale in dB)", fontsize=12)
    if cols_huge:
        df_numeric[cols_huge].boxplot(ax=axes[3])
        axes[3].set_title("Duration (Large Scale in ms)", fontsize=12)
        axes[3].ticklabel_format(style='plain', axis='y')

    plt.suptitle("Distribution of Features Grouped by Scale", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_scatter(df):
    """Plots scatter relationships for linear/non-linear comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(df['loudness'], df['energy'], alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Loudness')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Loudness vs Energy')
    
    axes[1].scatter(df['loudness'], df['popularity'], alpha=0.3, color='coral')
    axes[1].set_xlabel('Loudness')
    axes[1].set_ylabel('Popularity')
    axes[1].set_title('Loudness vs Popularity')
    
    axes[2].scatter(df['danceability'], df['popularity'], alpha=0.3, color='mediumseagreen')
    axes[2].set_xlabel('Danceability')
    axes[2].set_ylabel('Popularity')
    axes[2].set_title('Danceability vs Popularity')
    
    plt.suptitle("Scatter Plots: Physical trends vs Chaotic Popularity", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_histograms(df_numeric):
    df_numeric.hist(figsize=(15, 10), bins=30, edgecolor='black')
    plt.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X):
    """Plots the feature importances extracted from a tree-based model."""
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15) # Show top 15
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title("Top 15 Feature Importances (Decision Tree)")
    plt.tight_layout()
    plt.show()