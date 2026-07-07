import pytest
import numpy as np
from src.training.model_trainer import ModelTrainer
from src.evaluation.model_evaluator import ModelEvaluator

def test_model_trainer_initialization():
    trainer = ModelTrainer(model_type='rf')
    assert trainer.model is None
    assert trainer.model_type == 'rf'

def test_model_creation():
    trainer = ModelTrainer(model_type='rf')
    trainer.create_model()
    assert trainer.model is not None

def test_model_training():
    # Create sample data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    
    # Train Random Forest
    rf_trainer = ModelTrainer(model_type='rf')
    rf_trainer.create_model()
    rf_trainer.train(X, y)
    
    # Make predictions
    predictions = rf_trainer.model.predict(X)
    assert len(predictions) == len(y)

def test_model_evaluation():
    # Create sample data and predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Get metrics
    metrics = evaluator.evaluate_model(y_true, y_pred, "Test Model")
    
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics