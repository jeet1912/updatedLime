import random
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, tree, svm
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from sklearn import linear_model
import sklearn.metrics.pairwise
from sklearn.datasets import fetch_20newsgroups

POLARITY_PATH = '/content/drive/MyDrive/dm634/multi_domain_polarity/'
def LoadDataset(dataset_name):
    if dataset_name.endswith('ng'):
        if dataset_name == '2ng':
            cats = ['alt.atheism', 'soc.religion.christian']
            class_names = ['Atheism', 'Christianity']
        elif dataset_name == 'talkng':
            cats = ['talk.politics.guns', 'talk.politics.misc']
            class_names = ['Guns', 'PoliticalMisc']
        elif dataset_name == '3ng':
            cats = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.windows.x']
            class_names = ['windows.misc', 'ibm.hardware', 'windows.x']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)
        train_data = newsgroups_train.data
        train_labels = newsgroups_train.target
        test_data = newsgroups_test.data
        test_labels = newsgroups_test.target
        return train_data, train_labels, test_data, test_labels, class_names
    
    elif dataset_name.startswith('multi_polarity_'):
        name = dataset_name.split('_')[2]
        print('Name : ', name)
        return LoadMultiDomainDataset(POLARITY_PATH + name)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

def LoadMultiDomainDataset(path_data, remove_bigrams=True):
    random.seed(33)
    pos = []
    neg = []
    
    def get_words(line, remove_bigrams=True):
        z = [tuple(x.split(':')) for x in re.findall(r'\w+?:\d+', line)]
        if remove_bigrams:
            z = ' '.join([' '.join([x[0]] * int(x[1])) for x in z if '_' not in x[0]])
        else:
            z = ' '.join([' '.join([x[0]] * int(x[1])) for x in z])
        return z

    with open(os.path.join(path_data, 'negative.review'), 'r', encoding='iso-8859-1') as f:
        for line in f:
            neg.append(get_words(line, remove_bigrams))
    
    with open(os.path.join(path_data, 'positive.review'), 'r', encoding='iso-8859-1') as f:
        for line in f:
            pos.append(get_words(line, remove_bigrams))

    print('Len of pos ', len(np.array(pos)))
    print('Len of pos ', len(np.array(neg)))
    random.shuffle(pos)
    random.shuffle(neg)
    
    split_pos = int(len(pos) * 0.8)
    split_neg = int(len(neg) * 0.8)
    
    train_data = pos[:split_pos] + neg[:split_neg]
    test_data = pos[split_pos:] + neg[split_neg:]
    train_labels = [1] * len(pos[:split_pos]) + [0] * len(neg[:split_neg])
    test_labels = [1] * len(pos[split_pos:]) + [0] * len(neg[split_neg:])
    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels), ['neg', 'pos']
