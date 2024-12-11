import numpy as np
from sklearn import linear_model
import sklearn.metrics.pairwise
import scipy as sp

class RandomExplainer:
    """Explainer that randomly selects features."""

    def explain_instance(self, instance_vector, label, classifier_fn, num_features, dataset=None):
        """
        Explain an instance by randomly selecting features.

        Parameters:
        - instance_vector : numpy array, the instance to explain
        - label : int, the label for which to explain (unused in this case)
        - classifier_fn : callable, classifier's prediction function (unused here)
        - num_features : int, number of features to include in explanation
        - dataset : any, dataset info (unused here)

        Returns:
        - list of tuples : (feature index, weight), where weight is always 1
        """
        nonzero = instance_vector.nonzero()[1]
        explanation = np.random.choice(nonzero, size=min(num_features, len(nonzero)), replace=False)
        return [(x, 1.0) for x in explanation]

    def explain(self, train_vectors, train_labels, classifier, num_features, dataset):
        """
        Explain a random instance from the training set.

        Parameters:
        - train_vectors : numpy array, training data
        - train_labels : numpy array, training labels
        - classifier : sklearn classifier object
        - num_features : int, number of features for explanation
        - dataset : any, dataset info (unused here)

        Returns:
        - tuple : (instance index, explanation)
        """
        i = np.random.randint(0, train_vectors.shape[0])
        explanation = self.explain_instance(train_vectors[i], None, None, num_features, dataset)
        return i, explanation


class GeneralizedLocalExplainer:
    """Generalized local explainer using LIME principles."""

    def __init__(self, kernel_fn, data_labels_distances_mapping_fn, num_samples=5000, return_mean=True, verbose=False, return_mapped=False, positive=False):
        """
        Initialize the GeneralizedLocalExplainer.

        Parameters:
        - kernel_fn : callable, kernel function for weighting samples
        - data_labels_distances_mapping_fn : callable, function to generate neighborhood data
        - num_samples : int, number of samples in the neighborhood
        - return_mean : bool, whether to return the mean prediction
        - verbose : bool, whether to print verbose output
        - return_mapped : bool, whether to return feature names instead of indices
        - positive : bool, whether to enforce positive coefficients in LASSO
        """
        self.kernel_fn = kernel_fn
        self.data_labels_distances_mapping_fn = data_labels_distances_mapping_fn
        self.num_samples = num_samples
        self.return_mean = return_mean
        self.verbose = verbose
        self.return_mapped = return_mapped
        self.positive = positive

    def explain_instance(self, raw_data, label, classifier_fn, num_features, dataset=None):
        """
        Explain a single instance.

        Parameters:
        - raw_data : numpy array, instance to explain
        - label : int, label to explain
        - classifier_fn : callable, function to get prediction probabilities
        - num_features : int, number of features for explanation
        - dataset : any, dataset info (unused here)

        Returns:
        - list or tuple : explanation; if return_mean, returns (explanation, mean)
        """
        data, labels, distances, mapping = self.data_labels_distances_mapping_fn(raw_data, classifier_fn, self.num_samples)
        explanation, mean = self.explain_instance_with_data(data, labels, distances, label, num_features)
        
        if self.return_mapped:
            explanation = [(mapping[x[0]], x[1]) for x in explanation]
        if self.return_mean:
            return explanation, mean
        return explanation

    def explain_instance_with_data(self, data, labels, distances, label, num_features):
        """
        Explain instance using provided data.

        Parameters:
        - data : numpy array, neighborhood data
        - labels : numpy array, predictions for neighborhood data
        - distances : numpy array, distances from original instance
        - label : int, label to explain
        - num_features : int, number of features for explanation

        Returns:
        - tuple : (list of tuples for explanation, mean prediction)
        """
        weights = self.kernel_fn(distances)
        weighted_data = data * weights[:, np.newaxis]
        mean = np.mean(labels[:, label])
        shifted_labels = labels[:, label] - mean

        if self.verbose:
            print(f"Mean for label {label}: {mean}")

        weighted_labels = shifted_labels * weights
        lasso = linear_model.Lasso(alpha=1.0, fit_intercept=False, positive=self.positive)
        lasso.fit(weighted_data, weighted_labels)
        
        coef = lasso.coef_
        if sp.sparse.issparse(data):
            weighted_data = coef * data[0]
            feature_weights = sorted(zip(range(data.shape[1]), weighted_data), key=lambda x: np.abs(x[1]), reverse=True)
            explanation = feature_weights[:num_features]
        else:
            explanation = sorted(enumerate(coef), key=lambda x: np.abs(x[1]), reverse=True)[:num_features]

        return explanation, mean

def most_important_word(classifier, v, class_):
    """
    Find the word that has the most impact on the prediction for a given class.

    Parameters:
    - classifier : sklearn classifier with predict_proba method
    - v : numpy array, vector representation of the document
    - class_ : int, class index

    Returns:
    - int : index of the most impactful word or -1 if no positive impact found
    """
    orig_prob = classifier.predict_proba(v)[0][class_]
    max_change = 0
    max_index = -1
    for i in v.nonzero()[1]:  # only consider non-zero features
        original_value = v[0, i]
        v[0, i] = 0  # set feature to zero
        new_prob = classifier.predict_proba(v)[0][class_]
        change = orig_prob - new_prob
        if change > max_change:
            max_change = change
            max_index = i
        v[0, i] = original_value  # restore original value
    return max_index if max_change > 0 else -1

def explain_greedy(instance_vector, label, classifier_fn, num_features, dataset=None):
    """
    Greedy explanation by selecting features one by one.

    Parameters:
    - instance_vector : numpy array, instance to explain
    - label : int, label to explain
    - classifier_fn : callable, function to get prediction probabilities
    - num_features : int, number of features for explanation
    - dataset : any, dataset info (unused here)

    Returns:
    - list of tuples : (feature index, weight)
    """
    explanation = []
    for _ in range(num_features):
        word = most_important_word(classifier_fn, instance_vector.reshape(1, -1), label)
        if word == -1:
            break
        explanation.append((word, 1.0))
        instance_vector[0, word] = 0  # zero out selected word to see next impact
    return explanation
