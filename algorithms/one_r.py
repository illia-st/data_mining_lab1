import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class OneRClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.best_feature = None        # Індекс найкращої ознаки після навчання
        self.rules = {}                 # Правила: значення ознаки -> передбачений клас
        self.default_class = None       # Клас за замовчуванням (якщо зустрінеться нове значення ознаки)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        num_features = X.shape[1]              # Кількість ознак (стовпців)
        best_error = float('inf')             # Мінімальна кількість помилок (кращий результат)
        best_rules = None
        best_feature_index = None
        
        # Розрахувати загальний найбільш частий клас у датасеті (для default_class)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        overall_majority_class = unique_classes[np.argmax(class_counts)]
        self.default_class = overall_majority_class  # клас за замовчуванням (глобальна мода)
        
        # Перебір кожної ознаки, щоб знайти найкращу
        for fi in range(num_features):
            # Підрахунок частоти класів для кожного значення ознаки fi
            value_class_counts = {}  # словник: значення ознаки -> {клас -> кількість}
            
            # Отримуємо значення поточної ознаки
            feature_values = X[:, fi]
            
            # Для кожного унікального значення ознаки
            for value in np.unique(feature_values):
                hashable_value = value.item() if hasattr(value, 'item') else value
                
                # Знаходимо всі індекси де значення ознаки дорівнює поточному значенню
                indices = np.where(feature_values == value)[0]
                
                # Підраховуємо класи для цих індексів
                value_classes = y[indices]
                unique_value_classes, value_class_counts_arr = np.unique(value_classes, return_counts=True)
                
                # Зберігаємо підрахунок у словнику
                value_class_counts[hashable_value] = {
                    cls: count for cls, count in zip(unique_value_classes, value_class_counts_arr)
                }
            
            # Визначення правил для ознаки fi та підрахунок помилок
            rules = {}   # правила для поточної ознаки: значення -> прогнозований клас
            errors = 0   # кількість помилок при використанні цієї ознаки
            
            for value, class_count_map in value_class_counts.items():
                # Найбільш частий клас для даного значення ознаки (правило)
                best_class_for_value = max(class_count_map, key=class_count_map.get)
                rules[value] = best_class_for_value
                
                # Помилки: усі випадки цього значення, що не належать до best_class_for_value
                total_for_value = sum(class_count_map.values())
                errors_for_value = total_for_value - class_count_map[best_class_for_value]
                errors += errors_for_value
            
            # Перевірка, чи ця ознака краща (менше помилок)
            if errors < best_error:
                best_error = errors
                best_feature_index = fi
                best_rules = rules
        
        # Зберегти кращу ознаку та її правила
        self.best_feature = best_feature_index
        self.rules = best_rules
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        # Перевіряємо розмірність вхідних даних
        if len(X.shape) == 1:
            # Якщо це один приклад (одновимірний масив)
            value = X[self.best_feature]
            # Перетворюємо значення в хешований тип, якщо потрібно
            hashable_value = value.item() if hasattr(value, 'item') else value
            return self.rules.get(hashable_value, self.default_class)
        else:
            # Якщо це набір прикладів (двовимірний масив)
            predictions = []
            for row in X:
                value = row[self.best_feature]
                # Перетворюємо значення в хешований тип, якщо потрібно
                hashable_value = value.item() if hasattr(value, 'item') else value
                predicted_class = self.rules.get(hashable_value, self.default_class)
                predictions.append(predicted_class)
            return np.array(predictions)