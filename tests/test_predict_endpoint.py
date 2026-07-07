import json
import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

# Import the Flask app
from app import app, db, User
from werkzeug.security import generate_password_hash

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    with app.test_client() as client, app.app_context():
        db.create_all()  # Create tables
        yield client
        db.session.remove()
        db.drop_all()  # Clean up

@pytest.fixture
def test_user(client):
    user = User(
        username='testuser',
        email='test@example.com',
        password=generate_password_hash('password123')
    )
    db.session.add(user)
    db.session.commit()
    yield user

@pytest.fixture
def mock_model():
    mock = Mock()
    mock.predict.return_value = np.array([42.0])  # Constant prediction
    return mock

def test_predict_endpoint_returns_prediction(client, test_user, mock_model):
    # Set up mocks
    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[25.0, 60.0, 6.5, 80.0, 0.25, 0.15, 0.30]])

    # Mock label encoder for crop
    mock_encoder = Mock()
    mock_encoder.transform.return_value = np.array([0])  # 0 for 'maize'
    mock_encoders = {'crop': mock_encoder}

    def prepare_side_effect(*args, **kwargs):
        print(f"prepare_features called with args={args}, kwargs={kwargs}")
        # Return preprocessed features as expected
        df = pd.DataFrame(
            [[25.0, 60.0, 6.5, 80.0, 0.25, 0.15, 0.30]],
            columns=['temperature', 'humidity', 'ph', 'rainfall', 
                    'nitrogen', 'phosphorus', 'potassium'])
        return df, mock_scaler, mock_encoders

    # Set up patches
    prep_patch = patch('src.preprocessing.data_processor.prepare_features', 
                      side_effect=prepare_side_effect)
    model_patch = patch('app.model', mock_model)
    scaler_patch = patch('app.scaler', mock_scaler)
    encoders_patch = patch('app.encoders', mock_encoders)
    columns_patch = patch('app.feature_columns', 
                         ['temperature', 'humidity', 'ph', 'rainfall',
                          'nitrogen', 'phosphorus', 'potassium'])

    # Apply all patches
    with prep_patch, model_patch, scaler_patch, encoders_patch, columns_patch:
        # Login
        login_resp = client.post('/login', data={
            'email': 'test@example.com',
            'password': 'password123'
        }, follow_redirects=True)
        assert login_resp.status_code == 200

        # Test data
        payload = {
            'temperature': 25.0,
            'humidity': 60.0,
            'ph': 6.5,
            'rainfall': 80.0,
            'nitrogen': 0.25,
            'phosphorus': 0.15,
            'potassium': 0.30,
            'crop': 'maize'
        }

        # Make prediction request
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'success' in data
        assert data['success'] is True
        assert 'prediction' in data
        assert isinstance(data['prediction'], float)
        assert data['prediction'] == 42.0  # Our mock model always returns 42.0
