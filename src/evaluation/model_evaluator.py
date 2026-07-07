from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        """
        Initialize the model evaluator
        """
        self.metrics = {}
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Calculate regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        return self.metrics[model_name]
    
    def compare_models(self):
        """
        Compare different models based on their metrics
        """
        comparison_df = pd.DataFrame(self.metrics).round(4)
        return comparison_df
    
    def plot_predictions(self, y_true, y_pred, model_name):
        """
        Create scatter plot of predicted vs actual values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Actual vs Predicted Values')
        plt.tight_layout()
        return plt
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """
        Create residual plot
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residual Plot')
        plt.tight_layout()
        return plt