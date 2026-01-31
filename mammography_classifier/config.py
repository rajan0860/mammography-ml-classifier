"""
Configuration settings for the mammography classifier project.
"""

CONFIG = {
    'cv_folds': 10,
    'random_state': 1,
    'train_size': 0.75,
    'test_size': 0.25,
    'svm_c': 1.0,
    'rf_estimators': 10,
    'knn_default_n': 10,
    'knn_max_n': 50,
    'data_file': 'data/mammographic_masses.data.txt'
}

FEATURE_NAMES = ['age', 'shape', 'margin', 'density']
