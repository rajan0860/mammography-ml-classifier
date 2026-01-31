"""
Visualization and results reporting functions.
"""

from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pydotplus import graph_from_dot_data
from IPython.display import Image
from .config import CONFIG, FEATURE_NAMES


def visualize_decision_tree(training_inputs, testing_inputs, training_classes, testing_classes):
    """Train and visualize a decision tree classifier.
    
    Args:
        training_inputs: Training feature array
        testing_inputs: Testing feature array
        training_classes: Training target classes
        testing_classes: Testing target classes
    """
    dt_clf = DecisionTreeClassifier(random_state=CONFIG['random_state'])
    dt_clf.fit(training_inputs, training_classes)
    
    # Visualize the decision tree
    dot_data = StringIO()
    tree.export_graphviz(dt_clf, out_file=dot_data, feature_names=FEATURE_NAMES)
    graph = graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    
    accuracy = dt_clf.score(testing_inputs, testing_classes)
    print(f"\nDecision Tree - Test Set Accuracy: {accuracy:.4f}")


def print_results(results):
    """Print formatted results and highlight top performer.
    
    Args:
        results: Dictionary of classifier names to CV scores
    """
    print("\n" + "="*70)
    print("MODEL RANKING (Relative Performance)")
    print("="*70 + "\n")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    best_score = sorted_results[0][1]
    
    for rank, (model_name, score) in enumerate(sorted_results, 1):
        # Calculate percentage relative to best score
        percentage = (score / best_score) * 100
        
        # Create visual bar (20 chars max)
        bar_length = int(percentage / 5)  # Convert to 0-20 scale
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Print with rank, name, bar, percentage, and absolute score
        print(f"{rank:2d}. {model_name:<25} {bar} {percentage:5.1f}% ({score:.4f})")
    
    # Highlight top performer
    print("\n" + "="*70)
    print("🌟 TOP PERFORMER 🌟")
    print("="*70)
    
    best_name, best_score_val = sorted_results[0]
    worst_score = sorted_results[-1][1]
    improvement = ((best_score_val - worst_score) / worst_score) * 100
    
    print(f"  Model: {best_name.upper()}")
    print(f"  Score: {best_score_val:.4f}")
    print(f"  Improvement: {improvement:.1f}% better than worst model")
    print("="*70)
