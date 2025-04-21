import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_counts = {}        # Кількість прикладів кожного класу
        self.feature_value_counts = []# Список розміром num_features, 
                                      # де кожен елемент - словник {class -> {value -> count}}
        self.feature_values = []      # Список унікальних значень для кожної ознаки
        self.total_samples = 0        # Загальна кількість навчальних прикладів
        self.classes = []             # Список унікальних класів
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.total_samples = len(y)
        num_features = X.shape[1]
        
        # Підрахунок класів (апріорні)
        unique_classes, counts = np.unique(y, return_counts=True)
        self.classes = [cls.item() if hasattr(cls, 'item') else cls for cls in unique_classes]
        self.class_counts = {cls: count for cls, count in zip(self.classes, counts)}
        
        # Ініціалізація структур для підрахунку значень ознак
        self.feature_value_counts = [{cls: {} for cls in self.classes} for _ in range(num_features)]
        self.feature_values = [set() for _ in range(num_features)]
        
        # Підрахунок частот: скільки разів значення ознаки i зустрічається при класі cls
        for i in range(num_features):
            # Для кожної ознаки
            feature_col = X[:, i]
            
            # Знаходимо унікальні значення для цієї ознаки
            unique_values = np.unique(feature_col)
            
            # Додаємо унікальні значення (конвертовані в хешовані типи) до множини
            self.feature_values[i] = {val.item() if hasattr(val, 'item') else val for val in unique_values}
            
            # Для кожного класу підраховуємо частоти значень
            for cls in self.classes:
                # Знаходимо індекси, де клас == cls
                class_indices = np.where(y == cls)[0]
                
                # Значення ознаки для цих індексів
                class_feature_values = feature_col[class_indices]
                
                # Підрахунок частот унікальних значень
                unique_values_in_class, counts_in_class = np.unique(class_feature_values, return_counts=True)
                
                for val, count in zip(unique_values_in_class, counts_in_class):
                    hashable_val = val.item() if hasattr(val, 'item') else val
                    self.feature_value_counts[i][cls][hashable_val] = count
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        # Перевіряємо розмірність вхідних даних
        if len(X.shape) == 1:
            # Якщо це один приклад (одновимірний масив)
            return self._predict_single(X)
        else:
            # Якщо це набір прикладів (двовимірний масив)
            return np.array([self._predict_single(features) for features in X])
    
    def _predict_single(self, features):
        best_class = None
        best_log_prob = -math.inf
        
        for cls in self.classes:
            # log(P(C = cls))
            # Апріорна ймовірність P(C) = count(cls) / total_samples
            # Працюємо з логарифмами для зручності (уникнення дуже малих чисел)
            log_prob = math.log(self.class_counts[cls] / self.total_samples)
            
            # Для кожної ознаки додаємо log(P(feature = value | class = cls))
            for i, value in enumerate(features):
                hashable_value = value.item() if hasattr(value, 'item') else value
                
                # Кількість випадків, де ознака i = value при класі cls
                # Якщо такого значення при цьому класі не було, count = 0
                count = self.feature_value_counts[i][cls].get(hashable_value, 0)
                
                # Кількість різних значень ознаки i (для згладжування)
                N_values = len(self.feature_values[i])
                
                # Кількість прикладів класу cls
                class_count = self.class_counts[cls]
                
                prob = (count + 1) / (class_count + N_values)
                log_prob += math.log(prob)
            
            # Зберігаємо клас з найбільшою лог-імовірністю
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = cls

        return best_class