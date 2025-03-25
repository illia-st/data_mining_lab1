use std::collections::{HashMap, HashSet};

pub enum Node {
    Leaf(String),
    Decision {
        feature_index: usize,
        branches: HashMap<String, Box<Node>>,
    },
}

pub struct DecisionTreeClassifier {
    root: Node,
    default_class: String,  // глобальний клас за замовчуванням (наприклад, найбільш частий у навчанні)
}

impl DecisionTreeClassifier {
    pub fn new() -> Self {
        DecisionTreeClassifier {
            // Спочатку корінь можна тимчасово зробити листком з пустим класом
            root: Node::Leaf(String::new()),
            default_class: String::new(),
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<String>>, y: &Vec<String>) {
        // Обчислити глобальний переважний клас (для випадку непередбачених значень)
        let mut class_count = HashMap::new();
        for label in y {
            *class_count.entry(label.clone()).or_insert(0) += 1;
        }
        if let Some((majority_class, _)) = class_count.iter().max_by_key(|entry| entry.1) {
            self.default_class = majority_class.clone();
        }

        // Рекурсивно побудувати дерево
        let all_indices: Vec<usize> = (0..y.len()).collect();
        let feature_indices: Vec<usize> = (0..x[0].len()).collect();
        self.root = self.build_tree(x, y, &all_indices, &feature_indices);
    }

    pub fn build_tree(&self, x: &Vec<Vec<String>>, y: &Vec<String>, indices: &[usize], feature_indices: &[usize]) -> Node {
        // 1. Якщо всі приклади одного класу - повертати Leaf
        let first_class = &y[indices[0]];
        let all_same_class = indices.iter().all(|&i| &y[i] == first_class);
        if all_same_class {
            return Node::Leaf(first_class.clone());
        }

        // 2. Якщо не залишилось ознак - повернути Leaf з переважним класом цієї підмножини
        if feature_indices.is_empty() {
            // Знайти найчастіший клас серед indices
            let mut subset_class_count = HashMap::new();
            for &i in indices {
                *subset_class_count.entry(y[i].clone()).or_insert(0) += 1;
            }
            let majority_class = subset_class_count.into_iter().max_by_key(|entry| entry.1).unwrap().0;
            return Node::Leaf(majority_class);
        }

        // Функція для обчислення ентропії списку індексів
        let entropy = |idxs: &[usize]| {
            let total = idxs.len() as f64;
            let mut class_count = HashMap::new();
            for &i in idxs {
                *class_count.entry(y[i].clone()).or_insert(0_usize) += 1;
            }
            let mut ent = 0.0;
            for &count in class_count.values() {
                let p = count as f64 / total;
                if p > 0.0 {
                    ent -= p * p.log2();
                }
            }
            ent
        };

        let base_entropy = entropy(indices);

        // 3. Знайти ознаку з максимальним інформаційним приростом
        let mut best_feature = None;
        let mut best_info_gain = 0.0;
        let mut best_splits: HashMap<String, Vec<usize>> = HashMap::new();
        for &feature in feature_indices {
            // Розбити indices за значеннями ознаки feature
            let mut splits: HashMap<String, Vec<usize>> = HashMap::new();
            for &i in indices {
                let value = &x[i][feature];
                splits.entry(value.clone()).or_insert(Vec::new()).push(i);
            }
            // Обчислити ентропію після розбиття
            let mut new_entropy = 0.0;
            for subset_indices in splits.values() {
                if subset_indices.is_empty() { 
                    continue; 
                }
                let subset_entropy = entropy(subset_indices);
                new_entropy += (subset_indices.len() as f64 / indices.len() as f64) * subset_entropy;
            }
            let info_gain = base_entropy - new_entropy;
            if info_gain > best_info_gain {
                best_info_gain = info_gain;
                best_feature = Some(feature);
                best_splits = splits;
            }
        }

        // Якщо інформаційний приріст нульовий або не знайдено кращої ознаки – листок з переважним класом
        if best_feature.is_none() || best_info_gain <= 0.0 {
            let mut subset_class_count = HashMap::new();
            for &i in indices {
                *subset_class_count.entry(y[i].clone()).or_insert(0) += 1;
            }
            let majority_class = subset_class_count.into_iter().max_by_key(|entry| entry.1).unwrap().0;
            return Node::Leaf(majority_class);
        }

        let best_feature_idx = best_feature.unwrap();
        // 4. Рекурсивно побудувати гілки для кожного значення кращої ознаки
        let mut branches: HashMap<String, Box<Node>> = HashMap::new();
        // Сформувати список доступних ознак для дітей (виключаючи обрану)
        let mut remaining_features: Vec<usize> = feature_indices.to_vec();
        remaining_features.retain(|&f| f != best_feature_idx);
        for (value, subset_indices) in best_splits {
            if subset_indices.is_empty() {
                // Якщо жодного запису з таким значенням в цій підмножині,
                // створити листок з глобальним або локальним переважним класом
                branches.insert(value, Box::new(Node::Leaf(self.default_class.clone())));
            } else {
                // Рекурсивно викликати build_tree для підмножини
                let child_node = self.build_tree(x, y, &subset_indices, &remaining_features);
                branches.insert(value, Box::new(child_node));
            }
        }

        Node::Decision {
            feature_index: best_feature_idx,
            branches,
        }
    }

    pub fn predict(&self, data: &[Vec<String>]) -> Vec<String> {
        let mut predictions = Vec::new();
        for features in data {
            // Прогноз для одного прикладу
            let mut node = &self.root;
            let mut predicted_class = self.default_class.clone();
            loop {
                match node {
                    Node::Leaf(class) => {
                        predicted_class = class.clone();
                        break;
                    }
                    Node::Decision { feature_index, branches } => {
                        let feat_val = if *feature_index < features.len() {
                            &features[*feature_index]
                        } else {
                            // якщо не вистачає ознак у прикладі
                            predicted_class = self.default_class.clone();
                            break;
                        };
                        if let Some(next_node) = branches.get(feat_val) {
                            node = next_node;
                            continue;
                        } else {
                            // Якщо значення відсутнє серед гілок, використовуємо default_class і виходимо
                            predicted_class = self.default_class.clone();
                            break;
                        }
                    }
                }
            }
            predictions.push(predicted_class);
        }
        predictions
    }
}
