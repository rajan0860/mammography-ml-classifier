"""
Mammography ML Classifier - Main Script

This script evaluates multiple machine learning classifiers on mammographic
masses data to predict severity. It compares model performance and identifies
the best performing algorithm.

Package: mammography_classifier
Modules:
    - config.py: Configuration and constants
    - data.py: Data loading and preprocessing
    - models.py: Model definitions and evaluation
    - visualization.py: Results visualization and reporting
"""

from sklearn.model_selection import train_test_split
from mammography_classifier import (
    CONFIG,
    load_and_preprocess_data,
    get_minmax_scaled_features,
    define_classifiers,
    evaluate_all_classifiers,
    tune_knn,
    visualize_decision_tree,
    print_results,
)


def main():
    """Main execution function orchestrating the ML pipeline."""
    
    # Step 1: Load and preprocess data
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    all_features, all_features_scaled, all_classes = load_and_preprocess_data()
    
    # Step 2: Split data for decision tree visualization
    training_inputs, testing_inputs, training_classes, testing_classes = train_test_split(
        all_features_scaled, all_classes, 
        train_size=CONFIG['train_size'], 
        random_state=CONFIG['random_state']
    )
    
    # Step 3: Visualize decision tree
    print("\n" + "="*70)
    print("DECISION TREE VISUALIZATION")
    print("="*70)
    visualize_decision_tree(training_inputs, testing_inputs, training_classes, testing_classes)
    
    # Step 4: Prepare features with different scaling methods
    all_features_minmax = get_minmax_scaled_features(all_features)
    
    # Step 5: Define and evaluate classifiers
    classifiers, feature_map = define_classifiers(all_features_scaled, all_features_minmax)
    results = evaluate_all_classifiers(
        classifiers, feature_map, 
        all_features_scaled, all_features_minmax, 
        all_classes
    )
    
    # Step 6: Tune KNN hyperparameters
    tune_knn(all_features_scaled, all_classes)
    
    # Step 7: Print results summary
    print_results(results)


if __name__ == "__main__":
    main()