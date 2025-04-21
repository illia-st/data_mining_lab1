import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionTreeClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.tree = None  # дерево буде зберігатись як вкладені словники
        self.max_depth = max_depth  # максимальна глибина дерева
        self.min_samples_split = min_samples_split  # мінімальна кількість прикладів для розбиття
        self.default_class = None  # переважний клас для випадків відсутності значення

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Визначаємо переважний клас у всьому наборі даних
        unique_classes, counts = np.unique(y, return_counts=True)
        self.default_class = unique_classes[np.argmax(counts)]
        
        # Зберігаємо всі класи
        self.classes_ = unique_classes
        
        # Побудувати дерево рекурсивно, починаючи з усіх індексів прикладів і всіх доступних ознак
        features = list(range(X.shape[1]))  # індекси всіх ознак, які ще можна використовувати
        self.tree = self._build_tree(X, y, np.array(range(len(y))), features, depth=0)
        
        return self

    def _build_tree(self, X, y, indices, feature_indices, depth=0):
        # 1. Якщо всі приклади належать до одного класу - повернути листок
        current_classes = y[indices]
        unique_classes = np.unique(current_classes)
        
        if len(unique_classes) == 1:
            return {"label": unique_classes[0].item() if hasattr(unique_classes[0], 'item') else unique_classes[0]}
        
        # 2. Перевірка умов зупинки (max_depth, min_samples_split)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(indices) < self.min_samples_split or \
           len(feature_indices) == 0:
            # Повертаємо листок з переважним класом
            classes, counts = np.unique(current_classes, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            return {"label": majority_class.item() if hasattr(majority_class, 'item') else majority_class}

        # 3. Обчислити ентропію поточного вузла
        def entropy(idxs):
            classes_subset = y[idxs]
            _, counts = np.unique(classes_subset, return_counts=True)
            probabilities = counts / len(idxs)
            return -np.sum(probabilities * np.log2(probabilities))

        base_entropy = entropy(indices)

        # 4. Вибрати найкращу ознаку для розбиття (максимальний інформаційний приріст)
        best_feature = None
        best_info_gain = 0.0
        best_splits = None  # збережемо розбиття для найкращої ознаки
        
        for feature in feature_indices:
            # Розбити поточні дані за цією ознакою
            feature_values = X[indices, feature]
            unique_values = np.unique(feature_values)
            
            # Словник для зберігання розбиття
            splits = {}
            
            # Для кожного значення ознаки знаходимо відповідні індекси
            for value in unique_values:
                # Конвертуємо значення в хешований тип
                hashable_value = value.item() if hasattr(value, 'item') else value
                
                # Знаходимо індекси, де значення ознаки дорівнює поточному
                # Спочатку знаходимо маску для значень
                mask = feature_values == value
                # Потім застосовуємо маску до індексів
                value_indices = indices[mask]
                
                splits[hashable_value] = value_indices
            
            # Обчислити ентропію після розбиття (зважену суму ентропій підмножин)
            new_entropy = 0.0
            for subset_indices in splits.values():
                if len(subset_indices) == 0:
                    continue
                subset_entropy = entropy(subset_indices)
                new_entropy += (len(subset_indices) / len(indices)) * subset_entropy
                
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_splits = splits

        if best_feature is None or best_info_gain <= 0:
            # Якщо не вдалося знайти розбиття, повертаємо листок з переважним класом
            classes, counts = np.unique(current_classes, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            return {"label": majority_class.item() if hasattr(majority_class, 'item') else majority_class}

        # 5. Створити піддерева для кожного значення найкращої ознаки
        tree = {
            "feature": best_feature, 
            "branches": {},
            "majority_class": self.default_class.item() if hasattr(self.default_class, 'item') else self.default_class
        }
        
        # Зберігаємо переважний клас в цьому вузлі для випадків, коли зустрінеться нове значення
        classes, counts = np.unique(current_classes, return_counts=True)
        tree["majority_class"] = classes[np.argmax(counts)].item() if hasattr(classes[np.argmax(counts)], 'item') else classes[np.argmax(counts)]
        
        # Видалити використану ознаку зі списку доступних
        remaining_features = [f for f in feature_indices if f != best_feature]
        
        for value, subset_indices in best_splits.items():
            if len(subset_indices) == 0:
                # Якщо підмножина порожня, привласнити листок з переважним класом поточного вузла
                tree["branches"][value] = {"label": tree["majority_class"]}
            else:
                # Рекурсивно побудувати піддерево для підмножини
                tree["branches"][value] = self._build_tree(X, y, subset_indices, remaining_features, depth + 1)
                
        return tree

    def predict(self, X):
        X = np.asarray(X)
        
        if len(X.shape) == 1:
            # Один приклад
            return np.array([self._predict_single(X)])
        else:
            # Набір прикладів
            return np.array([self._predict_single(features) for features in X])

    def _predict_single(self, features):
        if self.tree is None:
            return self.default_class
            
        node = self.tree
        while "label" not in node:
            feature_idx = node["feature"]
            value = features[feature_idx]
            
            hashable_value = value.item() if hasattr(value, 'item') else value
            
            if hashable_value not in node["branches"]:
                return node["majority_class"]
                
            node = node["branches"][hashable_value]
            
        return node["label"]
    
   