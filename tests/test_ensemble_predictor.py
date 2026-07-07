"""
Tests for the ensemble model predictor utilities.
"""
import unittest
import numpy as np
from src.utils.ensemble_predictor import get_ensemble_prediction, load_ensemble_models
import joblib
import os

class TestEnsemblePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create dummy models for testing"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        
        # Create dummy models
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        svr = SVR(kernel='rbf')
        scaler = StandardScaler()
        
        # Dummy data
        X = np.random.rand(10, 7)  # 7 features
        y = np.random.rand(10)
        
        # Fit models
        scaled_X = scaler.fit_transform(X)
        rf.fit(scaled_X, y)
        svr.fit(scaled_X, y)
        
        # Save models
        if not os.path.exists('models/ensemble'):
            os.makedirs('models/ensemble')
        joblib.dump(rf, 'models/ensemble/rf_model.pkl')
        joblib.dump(svr, 'models/ensemble/svr_model.pkl')
        joblib.dump(scaler, 'models/ensemble/scaler.pkl')
    
    def test_load_models(self):
        """Test loading of ensemble models"""
        rf, svr, scaler = load_ensemble_models()
        self.assertIsNotNone(rf)
        self.assertIsNotNone(svr)
        self.assertIsNotNone(scaler)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        # Test input
        features = [25, 60, 6.5, 200, 50, 30, 20]
        
        # Get predictions
        pred = get_ensemble_prediction(features)
        self.assertIsNotNone(pred)
        self.assertIsInstance(pred, float)
        
        # Test individual predictions
        ensemble_pred, rf_pred, svr_pred = get_ensemble_prediction(features, return_individual=True)
        self.assertIsNotNone(ensemble_pred)
        self.assertIsNotNone(rf_pred)
        self.assertIsNotNone(svr_pred)
        
        # Verify ensemble is average
        self.assertAlmostEqual(ensemble_pred, (rf_pred + svr_pred) / 2)

if __name__ == '__main__':
    unittest.main()