import numpy as np
from Collections import Counter

class Node:
    def __init__(self, 
                 feature = None, 
                 threshold = None, 
                 left = None, 
                 right = None,
                 *,
                 value = None):
        
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None

    def is_leaf_node(self):
        return self.value is not None

    
class DecisionTree:
    def __init__(self,
                 min_samples_split = 2,
                 max_depth = 100,
                 n_features = None):
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    
    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels ==1 or  n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace = False)

        # find the best split
        best_feature,  best_thresh  = self._best_split(X, y, feat_idxs)

        # create child nodes

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                 # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy()

        # create children

        # calculate the weighted avg. entropy of children

        # calculate the IG


    def predict():