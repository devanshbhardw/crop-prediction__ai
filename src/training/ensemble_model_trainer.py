from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os

def train_ensemble():
    """
    Train both Random Forest and SVR models without affecting the existing model.
    Saves the models separately for optional use.
    """
    # Load the dataset
    data = pd.read_csv('data/sample_crop_data.csv')

    # Prepare features and target
    X = data[['temperature', 'humidity', 'ph', 'rainfall', 'nitrogen', 'phosphorus', 'potassium']]
    y = data['yield']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize both models
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)

    # Train both models
    print("Training Random Forest model...")
    rf_model.fit(X_train_scaled, y_train)
    
    print("Training SVR model...")
    svr_model.fit(X_train_scaled, y_train)

    # Make predictions with both models
    rf_pred = rf_model.predict(X_test_scaled)
    svr_pred = svr_model.predict(X_test_scaled)

    # Calculate ensemble predictions (average of both models)
    ensemble_pred = (rf_pred + svr_pred) / 2

    # Evaluate models
    print("\nModel Performance:")
    print("-----------------")
    print(f"Random Forest R² Score: {r2_score(y_test, rf_pred):.4f}")
    print(f"SVR R² Score: {r2_score(y_test, svr_pred):.4f}")
    print(f"Ensemble R² Score: {r2_score(y_test, ensemble_pred):.4f}")

    # Create models directory if it doesn't exist
    os.makedirs('models/ensemble', exist_ok=True)

    # Save models in a separate directory to avoid conflicts
    joblib.dump(rf_model, 'models/ensemble/rf_model.pkl')
    joblib.dump(svr_model, 'models/ensemble/svr_model.pkl')
    joblib.dump(scaler, 'models/ensemble/scaler.pkl')

    print("\nModels saved in 'models/ensemble' directory")
    print("You can now optionally use these models without affecting the existing system")

if __name__ == "__main__":
    train_ensemble()