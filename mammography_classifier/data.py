"""
Data loading and preprocessing functions.
"""

import pandas as pd
import numpy
from sklearn import preprocessing
from .config import CONFIG, FEATURE_NAMES


def load_and_preprocess_data():
    """Load and preprocess the mammographic masses dataset.
    
    Returns:
        tuple: (all_features, all_features_scaled, all_classes)
    """
    # Read data
    masses_data = pd.read_csv(
        CONFIG['data_file'], 
        na_values=['?'], 
        names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']
    )
    
    # Display rows with missing values
    missing_rows = masses_data.loc[
        (masses_data['age'].isnull()) |
        (masses_data['shape'].isnull()) |
        (masses_data['margin'].isnull()) |
        (masses_data['density'].isnull())
    ]
    
    if len(missing_rows) > 0:
        print("Rows with missing values:")
        print(missing_rows)
    
    # Remove missing values
    masses_data.dropna(inplace=True)
    print(f"\nDataset shape after removing NaN: {masses_data.shape}")
    print("\nDataset statistics:")
    print(masses_data.describe())
    
    # Extract features and target
    all_features = masses_data[FEATURE_NAMES].values
    all_classes = masses_data['severity'].values
    
    # Standardize features
    scaler = preprocessing.StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    numpy.random.seed(CONFIG['random_state'])
    
    return all_features, all_features_scaled, all_classes


def get_minmax_scaled_features(all_features):
    """Apply MinMax scaling to features (required for Naive Bayes).
    
    Args:
        all_features: Raw feature array
        
    Returns:
        array: MinMax scaled features
    """
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(all_features)
