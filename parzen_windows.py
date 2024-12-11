import numpy as np
import scipy as sp
import argparse
import evaluate_explanations
import sys
import xgboost
from sklearn import ensemble
from sklearn import neighbors
import embedding_forest
from sklearn import linear_model, svm, tree, metrics

def get_classifier(name, vectorizer):
    if name == 'logreg':
        return linear_model.LogisticRegression(fit_intercept=True)
    if name == 'random_forest':
        return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=10)
    if name == 'svm':
        return svm.SVC(probability=True, kernel='rbf', C=10, gamma=0.001)
    if name == 'tree':
        return tree.DecisionTreeClassifier(random_state=1)
    if name == 'neighbors':
        return neighbors.KNeighborsClassifier()
    if name == 'embforest':
        return embedding_forest.EmbeddingForest(vectorizer)

class ParzenWindowClassifier:
    def __init__(self):
        # Changed to use np.power for consistency with Python 3 syntax
        self.kernel = lambda x, sigma: np.array(np.exp(-.5 * np.power(x, 2).sum(axis=1) / sigma ** 2) / (np.sqrt(2 * np.pi * sigma ** 2))).flatten()

    def fit(self, X, y):
        self.X = X.toarray()
        self.y = y
        self.ones = y == 1
        self.zeros = y == 0

    def predict(self, x):
        b = sp.sparse.csr_matrix(x - self.X)
        pr = self.kernel(b, self.sigma)
        prob = sum(pr[self.ones]) / sum(pr)
        return int(prob > 0.5)

    def predict_proba(self, x):
        b = sp.sparse.csr_matrix(x - self.X)
        pr = self.kernel(b, self.sigma)
        prob = sum(pr[self.ones]) / sum(pr)
        return np.array([1 - prob, prob])

    def find_sigma(self, sigmas_to_try, cv_X, cv_y):
        self.sigma = sigmas_to_try[0]
        best_mistakes = 2**32 - 1
        best_sigma = self.sigma
        for sigma in sorted(sigmas_to_try):
            self.sigma = sigma
            preds = [self.predict(cv_X[i]) for i in range(cv_X.shape[0])]
            mistakes = sum(cv_y != np.array(preds))
            print(f"{sigma} {mistakes}")
            sys.stdout.flush()
            if mistakes < best_mistakes:
                best_mistakes = mistakes
                best_sigma = sigma
        print(f'Best sigma achieves {best_mistakes} mistakes. Disagreement={float(best_mistakes) / cv_X.shape[0]}')
        self.sigma = best_sigma

    def explain_instance(self, x, _, __,num_features,___=None):
        minus = self.X - x
        b = sp.sparse.csr_matrix(minus)
        ker = self.kernel(b, self.sigma)
        #ker = np.array([self.kernel(z, self.sigma) for z in b])
        times = np.multiply(minus, ker[:,np.newaxis])
        sumk_0= sum(ker[self.zeros])
        sumk_1= sum(ker[self.ones])
        sumt_0 = sum(times[self.zeros])
        sumt_1 = sum(times[self.ones])
        sumk_total = sumk_0 + sumk_1
        exp = (sumk_0 * sumt_1 - sumk_1 * sumt_0) / (self.sigma **2 * sumk_total ** 2)
        features = x.nonzero()[1]
        values = np.array(exp[0, x.nonzero()[1]])[0]
        return sorted(zip(features, values), key=lambda x:np.abs(x[1]), reverse=True)[:num_features]   
         
def main():
    parser = argparse.ArgumentParser(description='Visualize some stuff')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset name')
    parser.add_argument('--algorithm', '-a', type=str, required=True, help='algorithm_name')
    args = parser.parse_args()

    train_data, train_labels, test_data, test_labels, _ = LoadDataset(args.dataset)
    vectorizer = CountVectorizer(lowercase=False, binary=True)
    train_vectors = vectorizer.fit_transform(train_data)
    num_train = int(train_vectors.shape[0] * 0.8)
    indices = np.random.choice(range(train_vectors.shape[0]), train_vectors.shape[0], replace=False)
    train_v = train_vectors[indices[:num_train]]
    y_v = train_labels[indices[:num_train]]
    train_cv = train_vectors[indices[num_train:]]
    y_cv = train_labels[indices[num_train:]]
    print(f'train_size {train_v.shape[0]}')
    print(f'cv_size {train_cv.shape[0]}')
    classifier = get_classifier(args.algorithm, vectorizer)
    classifier.fit(train_v, y_v)
    print('train accuracy:', metrics.accuracy_score(y_v, classifier.predict(train_v)))
    print('cv accuracy:', metrics.accuracy_score(y_cv, classifier.predict(train_cv)))
    yhat_v = classifier.predict(train_v)
    yhat_cv = classifier.predict(train_cv)
    p = ParzenWindowClassifier()
    p.fit(train_v, yhat_v)
    p.find_sigma([0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], train_cv, yhat_cv)
    print(f'Best sigma: {p.sigma}')

if __name__ == "__main__":
    main()
