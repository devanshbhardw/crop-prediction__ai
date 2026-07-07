"""
Train models on the synthetic/sample dataset and save artifacts for the Flask app.

Creates:
 - models/crop_yield_model.pkl
 - models/scaler.pkl
 - models/encoders.pkl

Run:
    python scripts/train_on_synthetic.py --data data/synthetic_crop_data.csv
"""
import argparse
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
# Ensure project root is on sys.path so `src` package can be imported when running scripts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing.data_processor import prepare_features


def train_and_save(data_path, model_path='models/crop_yield_model.pkl'):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv(data_path)

    if 'yield' not in df.columns:
        raise RuntimeError("Dataset must contain a 'yield' column")

    X = df.drop(columns=['yield'])
    y = df['yield']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit feature transformers on training set
    X_train_proc, scaler, encoders = prepare_features(X_train, fit=True)
    X_test_proc, _, _ = prepare_features(X_test, scaler=scaler, encoders=encoders, fit=False)

    # Train Random Forest (fast default)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_proc, y_train)
    rf_preds = rf.predict(X_test_proc)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_rmse = rf_mse ** 0.5

    # Train SVR (may be slower)
    svr = SVR(kernel='rbf', C=1.0)
    svr.fit(X_train_proc, y_train)
    svr_preds = svr.predict(X_test_proc)
    svr_mse = mean_squared_error(y_test, svr_preds)
    svr_rmse = svr_mse ** 0.5

    # Choose best model (lower RMSE)
    best = rf if rf_rmse <= svr_rmse else svr
    chosen = 'random_forest' if best is rf else 'svr'

    joblib.dump(best, model_path)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    # Save feature column order so inference can reconstruct same input shape
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    # compute defaults: numeric means and categorical modes from training set
    defaults = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            defaults[col] = float(X_train[col].mean())
        else:
            # use mode if available, otherwise empty string
            mode_val = X_train[col].mode()
            defaults[col] = str(mode_val.iloc[0]) if not mode_val.empty else ''
    joblib.dump(defaults, 'models/feature_defaults.pkl')

    print(f"Trained models: RF RMSE={rf_rmse:.2f}, SVR RMSE={svr_rmse:.2f}. Saved best: {chosen}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_crop_data.csv', help='Path to CSV dataset')
    args = parser.parse_args()

    train_and_save(args.data)
