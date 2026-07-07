from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, model_type='rf'):
        """
        Initialize the model trainer
        Args:
            model_type (str): Type of model to train ('rf' for Random Forest, 'svr' for Support Vector Regressor)
        """
        self.model_type = model_type
        self.model = None
        
    def create_model(self):
        """
        Create the specified model with default parameters
        """
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        elif self.model_type == 'svr':
            self.model = SVR(
                kernel='rbf',
                C=1.0
            )
        else:
            raise ValueError("Unsupported model type")
            
    def train(self, X_train, y_train):
        """
        Train the model on the given data
        """
        if self.model is None:
            self.create_model()
        self.model.fit(X_train, y_train)
        
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Perform grid search for hyperparameter optimization
        """
        if self.model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'svr':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
    def save_model(self, filepath):
        """
        Save the trained model to disk
        """
        joblib.dump(self.model, filepath)
        
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk
        """
        return joblib.load(filepath)