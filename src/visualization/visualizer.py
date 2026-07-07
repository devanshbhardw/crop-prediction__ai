import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest model
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        return plt
    else:
        raise ValueError("Model doesn't support feature importance visualization")

def plot_correlation_matrix(data):
    """
    Plot correlation matrix of features
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt

def plot_prediction_distribution(predictions):
    """
    Plot distribution of predictions
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, kde=True)
    plt.title('Distribution of Predictions')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Count')
    plt.tight_layout()
    return plt