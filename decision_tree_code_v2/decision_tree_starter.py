from collections import Counter

import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
from pydot import graph_from_dot_data
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from random import randint
import math

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO implement entropy function
        d = {}
        n = len(y)
        for i in y:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1
        res = 0
        for i in d.keys():
            res -= d[i] * np.log2(d[i]/n)/n
        #print("Entro: ", res)
        return res
        

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        n = len(X)
        count = 0
        y1 = []
        y2 = []
        for i in range(n):
            if X[i] < thresh:
                count += 1
                y1.append(y[i])
            else:
                y2.append(y[i])
        y1 = np.array(y1)
        y2 = np.array(y2)
        res = DecisionTree.entropy(y) - (count/n * DecisionTree.entropy(y1) + (1 - count/n) * DecisionTree.entropy(y2))
        return res

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(
                    np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresh[i, :]
                ])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(
                X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for i in range(self.n):
            x_, y_ = [], []
            for _ in range(len(X)):
                ind = randint(0, len(X) - 1)
                x_.append(X[ind])
                y_.append(y[ind])
            x_ = np.array(x_)
            y_ = np.array(y_)
            self.decision_trees[i].fit(x_, y_)

        return self.decision_trees

    def predict(self, X):
        # TODO implement function
        preds = [tree.predict(X) for tree in self.decision_trees]
        res = []
        for j in range(X.shape[0]):
            d = {}
            for i in range(len(preds)):
                if preds[i][j] not in d:
                    d[preds[i][j]] = 1
                else:
                    d[preds[i][j]] += 1
            res.append(max(d, key = d.get))
        return np.array(res)

    

class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            self.params = {}
        else:
            self.params = params
        # TODO implement function
        self.m = m
        self.n = n
        self.decision_trees = [DecisionTreeClassifier(random_state=i, **self.params) for i in range(n)]
        self.p = [0 for i in range(n)]
    
    def fit(self, X, y):
        # TODO implement function
        
        for i in range(self.n):
            ind = np.random.choice(X.shape[1], self.m, replace=False)
            x = X[:, ind]
            self.p[i] = ind
            x_, y_ = [], []
            for _ in range(len(x)):
                ind = randint(0, len(x) - 1)
                x_.append(x[ind])
                y_.append(y[ind])
            x_ = np.array(x_)
            y_ = np.array(y_)
            self.decision_trees[i].fit(x_, y_)

        return self.decision_trees

    def predict(self, X):
        # TODO implement function
        preds = [self.decision_trees[i].predict(X[:, self.p[i]]) for i in range(self.n)]
        res = []
        for j in range(X.shape[0]):
            d = {}
            for i in range(len(preds)):
                if preds[i][j] not in d:
                    d[preds[i][j]] = 1
                else:
                    d[preds[i][j]] += 1
            res.append(max(d, key = d.get))
        return np.array(res)
    

def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) *
                 (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(c, num_splits=3):
    print("Cross validation", cross_val_score(c, X, y, cv=num_splits))
    if hasattr(c, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in c.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions):
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('predictions.csv', index_label='Id')

    # Now download the predictions.csv file to submit.`

if __name__ == "__main__":
    dataset = "spam"
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
    }
    N = 200

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    # TODO
    dt = DecisionTree(3, features)
    dt.fit(X,y)
    res = dt.predict(X)
    count = 0
    for i in range(len(X)):
        if res[i] == y[i]:
            count += 1
    print("dt train: ", count/len(X))
    #evaluate(dt)
    #print(repr(dt))

    # Basic decision tree
    print("\n\nPart (c): sklearn's decision tree")
    # Hint: Take a look at the imports!
    clf = DecisionTreeClassifier(max_depth= 3) # TODO
    # TODO
    # Visualizing the tree
    clf.fit(X,y)
    #out = io.StringIO()
    #export_graphviz(
        #clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    #graph = graph_from_dot_data(out.getvalue())
    #graph_from_dot_data(out.getvalue())[0].write_pdf("%s-basic-tree.pdf" % dataset)
    evaluate(clf)

    # Bagged trees
    print("\n\nPart (d-e): bagged trees")
    bt = BaggedTrees()
    bt.fit(X,y)
    res = bt.predict(X)
    count = 0
    for i in range(len(X)):
        if res[i] == y[i]:
            count += 1
    print("bt train: ", count/len(X))
    #print(bt.major())
    # TODO
    evaluate(bt)
    # Random forest
    print("\n\nPart (f-g): random forest")
    # TODO
    rf = RandomForest(m = int(math.sqrt(X.shape[1])))
    rf.fit(X,y)
    res = rf.predict(X)
    count = 0
    for i in range(len(X)):
        if res[i] == y[i]:
            count += 1
    print("rf train: ", count/len(X))
    #print(rf.major(features))
    evaluate(rf)
    # Generate csv file of predictions on test data
    # TODO

