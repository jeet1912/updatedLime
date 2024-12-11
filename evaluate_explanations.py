import sys
import copy
import argparse
import explainers
import parzen_windows
import numpy as np
import pickle
import sklearn
from load_datasets import *
from sklearn.metrics import accuracy_score

def get_tree_explanation(tree, v):
    t = tree.tree_
    nonzero = v.nonzero()[1]
    current = 0
    left_child = t.children_left[current]
    exp = set()
    while left_child != sklearn.tree._tree.TREE_LEAF:
        left_child = t.children_left[current]
        right_child = t.children_right[current]
        f = t.feature[current]
        if f in nonzero:
            exp.add(f)
        if v[0,f] < t.threshold[current]:
            current = left_child
        else:
            current = right_child
    return exp

class ExplanationEvaluator:
    def __init__(self, classifier_names=None):
        self.classifier_names = classifier_names
        if not self.classifier_names:
            self.classifier_names = ['l2logreg', 'tree']
        self.classifiers = {}

    def init_classifiers(self, dataset):
        self.classifiers[dataset] = {}
        for classifier in self.classifier_names:
            if classifier == 'l2logreg':
                try_cs = np.arange(0.1, 0, -0.01)  # Changed for Python 3.x float division
                for c in try_cs:
                    self.classifiers[dataset]['l2logreg'] = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=c)
                    self.classifiers[dataset]['l2logreg'].fit(self.train_vectors[dataset], self.train_labels[dataset])
                    coefs = self.classifiers[dataset]['l2logreg'].coef_[0]
                    lengths = [len(np.where(abs(coefs) > 1e-6)[0])]  # Count non-negligible coefficients
                    if max(lengths) <= 10:
                        break
            if classifier == 'tree':
                self.classifiers[dataset]['tree'] = tree.DecisionTreeClassifier(random_state=1)
                self.classifiers[dataset]['tree'].fit(self.train_vectors[dataset], self.train_labels[dataset])
                lengths = [len(get_tree_explanation(self.classifiers[dataset]['tree'], self.train_vectors[dataset][i])) for i in range(self.train_vectors[dataset].shape[0])]

    def load_datasets(self, dataset_names):
        self.train_data = {}
        self.train_labels = {}
        self.test_data = {}
        self.test_labels = {}
        for dataset in dataset_names:
            train_data, train_labels, test_data, test_labels, _ = LoadDataset(dataset)
            self.train_data[dataset] = train_data.tolist() if hasattr(train_data, 'tolist') else train_data
            self.train_labels[dataset] = train_labels.tolist() if hasattr(train_labels, 'tolist') else train_labels
            self.test_data[dataset] = test_data.tolist() if hasattr(test_data, 'tolist') else test_data
            self.test_labels[dataset] = test_labels.tolist() if hasattr(test_labels, 'tolist') else test_labels
            
    def vectorize_and_train(self):
        self.vectorizer = None
        self.train_vectors = None
        self.test_vectors = None
        self.inverse_vocabulary = None
        print('Vectorizing...', end='')  # Changed print for Python 3
        print(len(self.train_data))
        for d in self.train_data:
            print('What is d? ', d)
            self.vectorizer[d] = CountVectorizer(lowercase=False, binary=True)
            self.train_vectors[d] = self.vectorizer[d].fit_transform(self.train_data[d])
            self.test_vectors[d] = self.vectorizer[d].transform(self.test_data[d])
            terms = np.array(list(self.vectorizer[d].vocabulary_.keys()))
            indices = np.array(list(self.vectorizer[d].vocabulary_.values()))
            self.inverse_vocabulary[d] = terms[np.argsort(indices)]
        print('Done')
        print('Training...')
        for d in self.train_data:
            print(d)
            self.init_classifiers(d)
        print('Done')

    def measure_explanation_hability(self, explain_fn, max_examples=None):
        """Asks for explanations for all predictions in the train and test set, with
        budget = size of explanation. Returns two maps (train_results,
        test_results), from dataset to classifier to list of recalls"""
        budget = 10
        train_results = None
        test_results = None
        for d in self.train_data:
            train_results[d] = None
            test_results[d] = None
            print(f'Dataset: {d}')  # Changed to f-string
            for c in self.classifiers[d]:
                train_results[d][c] = None
                test_results[d][c] =None
                if c == 'l2logreg':
                    c_features = self.classifiers[d][c].coef_.nonzero()[1]
                print(f'classifier: {c}')  # Changed to f-string
                for i in range(len(self.test_data[d])):
                    if c == 'l2logreg':
                        true_features = set([x for x in self.test_vectors[d][i].nonzero()[1] if x in c_features])
                    elif c == 'tree':
                        true_features = get_tree_explanation(self.classifiers[d][c], self.test_vectors[d][i])
                    if len(true_features) == 0:
                        continue
                    to_get = budget
                    
                    # Ensure explain_fn can handle numpy arrays if self.test_vectors[d][i] or self.test_labels[d][i] are numpy arrays
                    exp_features = set(map(lambda x: x[0], explain_fn(self.test_vectors[d][i], self.test_labels[d][i], self.classifiers[d][c], to_get, d)))
                    
                    test_results[d][c].append(float(len(true_features.intersection(exp_features))) / len(true_features))
                    if max_examples and i >= max_examples:
                        break
        return train_results, test_results
        
def main():
    parser = argparse.ArgumentParser(description='Evaluate some explanations')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset name')
    parser.add_argument('--algorithm', '-a', type=str, required=True, help='algorithm_name')
    parser.add_argument('--explainer', '-e', type=str, required=True, help='explainer name')
    args = parser.parse_args()
    dataset = args.dataset
    algorithm = args.algorithm
    evaluator = ExplanationEvaluator(classifier_names=[algorithm])
    evaluator.load_datasets([dataset])
    evaluator.vectorize_and_train()
    explain_fn = None
    if args.explainer == 'lime':
        rho = 25
        kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
        explainer = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=False, verbose=False, return_mapped=True)
        explain_fn = explainer.explain_instance
    elif args.explainer == 'parzen':
        sigmas = {
            'multi_polarity_electronics': {'tree': 0.5, 'l2logreg': 1},
            'multi_polarity_kitchen': {'tree': 0.75, 'l2logreg': 2.0},
            'multi_polarity_dvd': {'tree': 8.0, 'l2logreg': 1},
            'multi_polarity_books': {'tree': 2.0, 'l2logreg': 2.0}
        }
        explainer = parzen_windows.ParzenWindowClassifier()
        cv_preds = sklearn.model_selection.cross_val_predict(evaluator.classifiers[dataset][algorithm], evaluator.train_vectors[dataset], evaluator.train_labels[dataset])  # Updated from cross_validation to model_selection
        explainer.fit(evaluator.train_vectors[dataset], cv_preds)
        explainer.sigma = sigmas[dataset][algorithm]
        explain_fn = explainer.explain_instance
    elif args.explainer == 'greedy':
        explain_fn = explainers.explain_greedy
    elif args.explainer == 'random':
        explainer = explainers.RandomExplainer()
        explain_fn = explainer.explain_instance
    train_results, test_results = evaluator.measure_explanation_hability(explain_fn)
    average_test = np.nanmean(test_results[dataset][algorithm])
    print(f'Average test: {average_test}')  # Changed to f-string
    out = {'train': train_results[dataset][algorithm], 'test': test_results[dataset][algorithm]}

if __name__ == "__main__":
    main()




