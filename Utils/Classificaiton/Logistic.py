import numpy as np



class LogisticRegressionClassifier:
    def __init__(self,
                 lr = 0.001, 
                 n_iters = 1000):
        
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weithts = np.zeros(self.features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        output = [0 if y <= 0.5 else 1 for y in y_pred]
        return output

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))