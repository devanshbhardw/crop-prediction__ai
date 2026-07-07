"""
Optional utility functions for using the ensemble model prediction.
This can be imported and used separately without affecting the main application.
"""
import joblib
import numpy as np

def load_ensemble_models():
    """Load both RF and SVR models"""
    try:
        rf_model = joblib.load('models/ensemble/rf_model.pkl')
        svr_model = joblib.load('models/ensemble/svr_model.pkl')
        scaler = joblib.load('models/ensemble/scaler.pkl')
        return rf_model, svr_model, scaler
    except:
        print("Ensemble models not found. Please run ensemble_model_trainer.py first")
        return None, None, None

def get_ensemble_prediction(features, return_individual=False):
    """
    Get prediction using both models.
    
    Args:
        features: List of features [temperature, humidity, ph, rainfall, nitrogen, phosphorus, potassium]
        return_individual: If True, returns individual model predictions along with ensemble
    
    Returns:
        If return_individual=False: ensemble prediction
        If return_individual=True: (ensemble_prediction, rf_prediction, svr_prediction)
    """
    rf_model, svr_model, scaler = load_ensemble_models()
    if not all([rf_model, svr_model, scaler]):
        return None
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Get predictions
    rf_pred = rf_model.predict(scaled_features)[0]
    svr_pred = svr_model.predict(scaled_features)[0]
    
    # Calculate ensemble prediction
    ensemble_pred = (rf_pred + svr_pred) / 2
    
    if return_individual:
        return ensemble_pred, rf_pred, svr_pred
    return ensemble_pred