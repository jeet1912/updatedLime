import sys
import copy
import os
import numpy as np
import scipy as sp
import json
import random
import sklearn
from sklearn import ensemble, svm, tree, neighbors, linear_model
import pickle
import explainers
import parzen_windows
import embedding_forest
from load_datasets import LoadDataset
import argparse
import collections

def get_classifier(name, vectorizer):
    """Returns an initialized classifier based on the name provided."""
    if name == 'logreg':
        return linear_model.LogisticRegression(fit_intercept=True)
    elif name == 'random_forest':
        return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=-1)
    elif name == 'svm':
        return svm.SVC(probability=True, kernel='rbf', C=10, gamma='scale')  # 'scale' is the recommended default for gamma
    elif name == 'tree':
        return tree.DecisionTreeClassifier(random_state=1)
    elif name == 'neighbors':
        return neighbors.KNeighborsClassifier()
    elif name == 'embforest':
        return embedding_forest.EmbeddingForest(vectorizer)
    raise ValueError(f"Unknown classifier: {name}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate explanations for trustworthiness')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--algorithm', '-a', type=str, required=True, help='Algorithm name')
    parser.add_argument('--num_features', '-k', type=int, required=True, help='Number of features')
    parser.add_argument('--percent_untrustworthy', '-u', type=float, required=True, help='Percentage of untrustworthy features')
    parser.add_argument('--num_rounds', '-r', type=int, required=True, help='Number of rounds')
    args = parser.parse_args()

    dataset = args.dataset
    train_data, train_labels, test_data, test_labels, class_names = LoadDataset(dataset)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(lowercase=False, binary=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    
    clf = get_classifier(args.algorithm, vectorizer)
    clf.fit(train_vectors, train_labels)
    
    explainer_names = ['LIME', 'parzen', 'random', 'greedy']
    precision = collections.defaultdict(list)
    recall = collections.defaultdict(list)
    f1 = collections.defaultdict(list)
    
    for _ in range(args.num_rounds):
        flipped_preds_size = []
        untrustworthy = random.sample(list(range(train_vectors.shape[1])), int(args.percent_untrustworthy * train_vectors.shape[1]))
        
        exps = {
            'LIME': [],
            'parzen': [],
            'random': [],
            'greedy': []
        }
        
        for i in range(test_vectors.shape[0]):
            # LIME
            exp, mean = explainers.explain_instance(test_vectors[i], 1, clf.predict_proba, args.num_features)
            exps['LIME'].append((exp, mean))
            
            # Parzen
            exp = parzen_windows.explain_instance(test_vectors[i], 1, clf.predict_proba, args.num_features)
            mean = parzen_windows.predict_proba(test_vectors[i])[1]
            exps['parzen'].append((exp, mean))
            
            # Random
            exp = explainers.RandomExplainer().explain_instance(test_vectors[i], 1, None, args.num_features)
            exps['random'].append(exp)
            
            # Greedy
            exp = explainers.explain_greedy(test_vectors[i], test_labels[i], clf.predict_proba, args.num_features)
            exps['greedy'].append(exp)

        trust = collections.defaultdict(set)
        mistrust = collections.defaultdict(set)
        trust_fn = lambda prev, curr: (prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5)
        trust_fn_all = lambda exp, unt: len([x[0] for x in exp if x[0] in unt]) == 0
        shouldnt_trust = set()

        for i in range(test_vectors.shape[0]):
            original_pred = clf.predict_proba(test_vectors[i])[0][1]
            modified_pred = original_pred - sum([x[1] for x in exps['LIME'][i][0] if x[0] in untrustworthy])
            if not trust_fn(original_pred, modified_pred):
                shouldnt_trust.add(i)
                flipped = abs(original_pred - modified_pred)
                flipped_preds_size.append(flipped)

            for expl in explainer_names:
                if expl == 'LIME':
                    exp, mean = exps['LIME'][i]
                    prev_tot = mean
                elif expl == 'parzen':
                    exp, mean = exps['parzen'][i]
                    prev_tot = mean
                elif expl == 'random':
                    exp = exps['random'][i]
                else:  # greedy
                    exp = exps['greedy'][i]

                if expl in ['LIME', 'parzen']:
                    tot = prev_tot - sum([x[1] for x in exp if x[0] in untrustworthy])
                    trust[expl].add(i) if trust_fn(tot, prev_tot) else mistrust[expl].add(i)
                else:  # random, greedy
                    trust[expl].add(i) if trust_fn_all(exp, untrustworthy) else mistrust[expl].add(i)

        for expl in explainer_names:
            false_positives = trust[expl].intersection(shouldnt_trust)
            true_positives = trust[expl].difference(shouldnt_trust)
            false_negatives = mistrust[expl].difference(shouldnt_trust)
            true_negatives = mistrust[expl].intersection(shouldnt_trust)

            try:
                prec = len(true_positives) / (len(true_positives) + len(false_positives))
            except ZeroDivisionError:
                prec = 0
            try:
                rec = len(true_positives) / (len(true_positives) + len(false_negatives))
            except ZeroDivisionError:
                rec = 0
            precision[expl].append(prec)
            recall[expl].append(rec)
            f1_score = 2 * (prec * rec) / (prec + rec) if (prec and rec) else 0
            f1[expl].append(f1_score)

    print('Average number of flipped predictions:', np.mean(flipped_preds_size), '+-', np.std(flipped_preds_size))
    print('Precision:')
    for expl in explainer_names:
        print(f"{expl}: {np.mean(precision[expl]):.4f} +- {np.std(precision[expl]):.4f}, p-value {sp.stats.ttest_ind(precision[expl], precision['LIME'])[1]:.4f}")
    print('\nRecall:')
    for expl in explainer_names:
        print(f"{expl}: {np.mean(recall[expl]):.4f} +- {np.std(recall[expl]):.4f}, p-value {sp.stats.ttest_ind(recall[expl], recall['LIME'])[1]:.4f}")
    print('\nF1:')
    for expl in explainer_names:
        print(f"{expl}: {np.mean(f1[expl]):.4f} +- {np.std(f1[expl]):.4f}, p-value {sp.stats.ttest_ind(f1[expl], f1['LIME'])[1]:.4f}")

if __name__ == "__main__":
    main()
