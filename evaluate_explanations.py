import sys
import numpy as np
import scipy as sp
import collections
import argparse
import pickle
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

def load_explanations(filename):
    """
    Load explanations from a pickle file.

    Parameters:
    - filename : str, path to the pickle file

    Returns:
    - dict : loaded explanations
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def evaluate_explanations(explanations, classifier, test_data, test_labels, num_features=None):
    """
    Evaluate the explanations by comparing the original and perturbed predictions.

    Parameters:
    - explanations : dict, {'explainer_name': [(explanation, mean)]}
    - classifier : sklearn classifier object
    - test_data : numpy array, test data
    - test_labels : numpy array, true labels
    - num_features : int or None, number of features to consider in evaluation

    Returns:
    - dict : performance metrics for each explainer
    """
    results = collections.defaultdict(list)
    for explainer, exps in explanations.items():
        for instance, (explanation, mean) in enumerate(exps):
            original_pred = classifier.predict_proba([test_data[instance]])[0]
            if num_features:
                explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:num_features]
            
            # Perturb the instance by zeroing out features in the explanation
            perturbed_instance = test_data[instance].copy()
            for feature, _ in explanation:
                perturbed_instance[feature] = 0
            
            perturbed_pred = classifier.predict_proba([perturbed_instance])[0]
            change = original_pred - perturbed_pred
            
            # Compute metrics
            accuracy = metrics.accuracy_score([test_labels[instance]], [np.argmax(original_pred)])
            fidelity = 1 - np.mean(np.abs(change))
            results[explainer].append({
                'accuracy': accuracy,
                'fidelity': fidelity,
                'change': change
            })

    # Aggregate results
    aggregated_results = {}
    for explainer, metrics_list in results.items():
        aggregated_results[explainer] = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'fidelity': np.mean([m['fidelity'] for m in metrics_list]),
            'change': np.mean([np.abs(m['change']).sum() for m in metrics_list])
        }
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate the quality of explanations.')
    parser.add_argument('--explanation_file', '-e', type=str, required=True, help='File containing explanations')
    parser.add_argument('--classifier_file', '-c', type=str, required=True, help='File containing the trained classifier')
    parser.add_argument('--test_data_file', '-t', type=str, required=True, help='File containing the test data')
    parser.add_argument('--num_features', '-n', type=int, default=None, help='Number of top features to consider in evaluation')
    args = parser.parse_args()

    # Load data
    explanations = load_explanations(args.explanation_file)
    with open(args.classifier_file, 'rb') as f:
        classifier = pickle.load(f)
    with open(args.test_data_file, 'rb') as f:
        test_data, test_labels = pickle.load(f)

    # Evaluate explanations
    results = evaluate_explanations(explanations, classifier, test_data, test_labels, args.num_features)

    # Print results
    for explainer, metrics in results.items():
        print(f"\nExplainer: {explainer}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
