import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """
    Load a CSV dataset into a DataFrame
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """
    Clean the dataset by handling missing values and outliers
    """
    # Handle missing numeric values by column mean
    data = data.copy()
    data = data.fillna(data.mean(numeric_only=True))
    return data


def feature_engineering(data):
    """
    Create new features and transform existing ones (placeholder)
    """
    # Add feature engineering steps here as needed
    return data


def prepare_features(data, scaler=None, encoders=None, fit=True):
    """
    Prepare features for model training or inference.

    Args:
        data (pd.DataFrame): input feature dataframe (should NOT contain target column)
        scaler (StandardScaler or None): existing scaler to use when fit=False
        encoders (dict or None): mapping of {feature_name: LabelEncoder} to use when fit=False
        fit (bool): whether to fit scaler/encoders on the provided data. If False, scaler and
                    encoders must be provided and will be used to transform.

    Returns:
        data_transformed (pd.DataFrame): transformed feature dataframe (numpy array safe)
        scaler (StandardScaler): fitted scaler (if fit=True) or the passed scaler
        encoders (dict): dict of fitted LabelEncoders for categorical features
    """
    df = data.copy()

    # Select numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Scaler
    if fit:
        scaler = StandardScaler()
        if len(numeric_features) > 0:
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
    else:
        if scaler is None and len(numeric_features) > 0:
            raise ValueError("scaler must be provided when fit=False and numeric features exist")
        if len(numeric_features) > 0:
            df[numeric_features] = scaler.transform(df[numeric_features])

    # Encoders: use one LabelEncoder per categorical column
    if fit:
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        if encoders is None:
            # If there are no categorical features this is okay; otherwise raise
            if len(categorical_features) > 0:
                raise ValueError("encoders must be provided when fit=False and categorical features exist")
        else:
            for col in categorical_features:
                if col not in encoders:
                    raise ValueError(f"Missing encoder for column: {col}")
                le = encoders[col]
                df[col] = le.transform(df[col].astype(str))

    return df, scaler, encoders