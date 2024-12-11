import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gensim.models import Word2Vec

class EmbeddingForest(BaseEstimator, ClassifierMixin):
    """
    An ensemble of decision trees where each tree is trained on word embeddings.
    
    This class extends scikit-learn's classifier interface for use with word embeddings.
    """

    def __init__(self, vectorizer, n_estimators=10, max_depth=None, random_state=None):
        self.vectorizer = vectorizer
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        """
        Fit the Embedding Forest model.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
        - y : array-like of shape (n_samples,)

        Returns:
        - self : object
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Fit Word2Vec model if not provided
        if not hasattr(self.vectorizer, 'wv'):
            sentences = [self.vectorizer.inverse_transform([X[i]])[0] for i in range(X.shape[0])]
            self.vectorizer = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

        for _ in range(self.n_estimators):
            tree = tree.DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            
            # Create a new dataset by averaging word embeddings for each document
            X_embedded = np.array([np.mean([self.vectorizer.wv[word] for word in doc if word in self.vectorizer.wv] or [np.zeros(self.vectorizer.vector_size)], axis=0) for doc in self._decode(X)])
            
            tree.fit(X_embedded, y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        - X : array-like of shape (n_samples, n_features)

        Returns:
        - y : array of shape (n_samples,)
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        if not hasattr(self.vectorizer, 'wv'):
            raise ValueError("Word2Vec model not fitted. Please fit the model first.")

        # Convert text to embeddings for prediction
        X_embedded = np.array([np.mean([self.vectorizer.wv[word] for word in doc if word in self.vectorizer.wv] or [np.zeros(self.vectorizer.vector_size)], axis=0) for doc in self._decode(X)])

        predictions = np.array([tree.predict(X_embedded) for tree in self.estimators_])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        - X : array-like of shape (n_samples, n_features)

        Returns:
        - T : array of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        if not hasattr(self.vectorizer, 'wv'):
            raise ValueError("Word2Vec model not fitted. Please fit the model first.")

        # Convert text to embeddings for probability prediction
        X_embedded = np.array([np.mean([self.vectorizer.wv[word] for word in doc if word in self.vectorizer.wv] or [np.zeros(self.vectorizer.vector_size)], axis=0) for doc in self._decode(X)])

        probas = np.array([tree.predict_proba(X_embedded) for tree in self.estimators_])
        return np.mean(probas, axis=0)

    def _decode(self, X):
        """
        Convert sparse matrix to list of documents for embedding lookup.
        
        Parameters:
        - X : sparse matrix

        Returns:
        - list of lists of words
        """
        return [self.vectorizer.inverse_transform([X[i]])[0] for i in range(X.shape[0])]
