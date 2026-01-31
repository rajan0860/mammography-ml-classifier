"""
Model definition and evaluation functions.
"""

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from .config import CONFIG


def evaluate_classifier(clf, features, classes, name):
    """Evaluate a classifier using cross-validation.
    
    Args:
        clf: Classifier instance
        features: Feature array
        classes: Target class array
        name: Name of classifier for printing
        
    Returns:
        float: Mean cross-validation score
    """
    cv_scores = cross_val_score(clf, features, classes, cv=CONFIG['cv_folds'])
    mean_score = cv_scores.mean()
    print(f"{name}: {mean_score:.4f}")
    return mean_score


def define_classifiers(all_features_scaled, all_features_minmax):
    """Define all classifiers to be evaluated.
    
    Args:
        all_features_scaled: StandardScaler features
        all_features_minmax: MinMaxScaler features
        
    Returns:
        tuple: (classifiers dict, feature_map dict)
    """
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=CONFIG['random_state']),
        'Random Forest': RandomForestClassifier(
            n_estimators=CONFIG['rf_estimators'], 
            random_state=CONFIG['random_state']
        ),
        'SVM (Linear)': svm.SVC(kernel='linear', C=CONFIG['svm_c']),
        'SVM (RBF)': svm.SVC(kernel='rbf', C=CONFIG['svm_c']),
        'SVM (Sigmoid)': svm.SVC(kernel='sigmoid', C=CONFIG['svm_c']),
        'SVM (Poly)': svm.SVC(kernel='poly', C=CONFIG['svm_c']),
        'KNN (n=10)': neighbors.KNeighborsClassifier(n_neighbors=CONFIG['knn_default_n']),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
    }
    
    # Map feature scaling for each classifier
    feature_map = {
        'Naive Bayes': all_features_minmax,  # NB requires MinMax scaling
    }
    
    return classifiers, feature_map


def evaluate_all_classifiers(classifiers, feature_map, all_features_scaled, all_features_minmax, all_classes):
    """Evaluate all classifiers and return results.
    
    Args:
        classifiers: Dictionary of classifier instances
        feature_map: Dictionary mapping classifier names to their features
        all_features_scaled: StandardScaler features
        all_features_minmax: MinMaxScaler features
        all_classes: Target classes
        
    Returns:
        dict: Results mapping classifier names to CV scores
    """
    results = {}
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (10-Fold CV)")
    print("="*60 + "\n")
    
    for name, clf in classifiers.items():
        # Use appropriate feature scaling for each classifier
        features = feature_map.get(name, all_features_scaled)
        results[name] = evaluate_classifier(clf, features, all_classes, name)
    
    return results


def tune_knn(all_features_scaled, all_classes):
    """Tune KNN hyperparameter n_neighbors.
    
    Args:
        all_features_scaled: StandardScaler features
        all_classes: Target classes
        
    Returns:
        tuple: (best_k, best_score)
    """
    print("\n" + "="*60)
    print("KNN HYPERPARAMETER TUNING (n_neighbors from 1 to 49)")
    print("="*60 + "\n")
    
    best_k = 1
    best_score = 0
    
    for n in range(1, CONFIG['knn_max_n']):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n)
        cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=CONFIG['cv_folds'])
        mean_score = cv_scores.mean()
        print(f"n_neighbors={n}: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = n
    
    print(f"\nBest KNN - n_neighbors={best_k}, CV Score: {best_score:.4f}")
    return best_k, best_score
