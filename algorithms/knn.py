import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            return self._predict_one(X)
        else:
            return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x):
        distances = []
                
        for i, xi in enumerate(self.X_train):
            dist = np.sqrt(np.sum((xi - x) ** 2))
            distances.append((dist, self.y_train[i]))
        
        distances.sort(key=lambda d: d[0])
        nearest_neighbors = distances[:self.k]
        
        class_votes = {}
        for _, neighbor_class in nearest_neighbors:
            class_votes[neighbor_class] = class_votes.get(neighbor_class, 0) + 1
        
        predicted_class = max(class_votes, key=class_votes.get)
        return predicted_class