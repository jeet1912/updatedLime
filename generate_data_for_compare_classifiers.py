import sys
import copy
sys.path.append('..')
import time
import numpy as np
import scipy as sp
import sklearn
import xgboost
import xgboost.sklearn as xgb_sklearn  # Changed for explicitness, although not strictly necessary
import explainers
from load_datasets import *
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn.model_selection import cross_val_predict  # Changed from cross_validation to model_selection
import pickle
import parzen_windows
import argparse

def get_random_indices(labels, class_, probability):
    nonzero = np.where(labels == class_)[0]  # Changed to use np.where for Python 3 compatibility
    if len(nonzero) == 0 or probability == 0:
        return []
    return np.random.choice(nonzero, int(probability * len(nonzero)), replace=False)

def add_corrupt_feature(feature_name, clean_train, clean_test, dirty_train,
                        train_labels, test_labels, class_probs_dirty, class_probs_clean, fake_prefix='FAKE'):
    """clean_train, clean_test, dirty_train will be corrupted"""
    for class_ in set(train_labels):
        indices = get_random_indices(train_labels, class_, class_probs_clean[class_])
        for i in indices:
            clean_train[i] += f' {fake_prefix}{feature_name}{fake_prefix}'
        indices = get_random_indices(train_labels, class_, class_probs_dirty[class_])
        for i in indices:
            dirty_train[i] += f' {fake_prefix}{feature_name}{fake_prefix}'
        indices = get_random_indices(test_labels, class_, class_probs_clean[class_])
        for i in indices:
            clean_test[i] += f' {fake_prefix}{feature_name}{fake_prefix}'
