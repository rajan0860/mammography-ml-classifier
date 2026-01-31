"""
Mammography ML Classifier Package

A machine learning pipeline for classifying mammographic masses.
"""

__version__ = "1.0.0"
__author__ = "Rajan Mehta"

from .config import CONFIG, FEATURE_NAMES
from .data import load_and_preprocess_data, get_minmax_scaled_features
from .models import (
    evaluate_classifier,
    define_classifiers,
    evaluate_all_classifiers,
    tune_knn
)
from .visualization import visualize_decision_tree, print_results

__all__ = [
    'CONFIG',
    'FEATURE_NAMES',
    'load_and_preprocess_data',
    'get_minmax_scaled_features',
    'evaluate_classifier',
    'define_classifiers',
    'evaluate_all_classifiers',
    'tune_knn',
    'visualize_decision_tree',
    'print_results',
]
