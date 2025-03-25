use std::collections::{HashMap, HashSet};

pub struct NaiveBayesClassifier {
    class_counts: HashMap<String, usize>,
    feature_value_counts: Vec<HashMap<String, HashMap<String, usize>>>,
    feature_values: Vec<HashSet<String>>,
    classes: Vec<String>,
    total_samples: usize,
}

impl NaiveBayesClassifier {
    pub fn new() -> Self {
        NaiveBayesClassifier {
            class_counts: HashMap::new(),
            feature_value_counts: Vec::new(),
            feature_values: Vec::new(),
            classes: Vec::new(),
            total_samples: 0,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<String>>, y: &Vec<String>) {
        self.total_samples = y.len();
        let num_features = x[0].len();

        // 1. Підрахувати кількість прикладів для кожного класу
        for label in y {
            *self.class_counts.entry(label.clone()).or_insert(0) += 1;
        }
        // Зберегти список класів
        self.classes = self.class_counts.keys().cloned().collect();

        // 2. Ініціалізувати структури для підрахунку значень ознак
        self.feature_value_counts = vec![HashMap::new(); num_features];
        self.feature_values = vec![HashSet::new(); num_features];
        for i in 0..num_features {
            // Для кожної ознаки підготувати карту для кожного класу
            for cls in &self.classes {
                self.feature_value_counts[i].insert(cls.clone(), HashMap::new());
            }
        }

        // 3. Підрахувати частоти значень ознак при кожному класі
        for (features, label) in x.iter().zip(y.iter()) {
            for (i, value) in features.iter().enumerate() {
                // Додати значення в множину можливих значень ознаки i
                self.feature_values[i].insert(value.clone());
                // Збільшити лічильник для (клас = label, ознака i = value)
                if let Some(class_map) = self.feature_value_counts[i].get_mut(label) {
                    *class_map.entry(value.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn predict(&self, data: &[Vec<String>]) -> Vec<String> {
        // Прогноз класів для списку прикладів
        let mut predictions = Vec::new();
        for features in data {
            let mut best_class = String::new();
            let mut best_log_prob = f64::MIN;
            // Обчислити "скори" (log-імовірності) для кожного класу
            for cls in &self.classes {
                // log(P(C = cls))
                let class_count = *self.class_counts.get(cls).unwrap_or(&0) as f64;
                // Якщо class_count = 0 (теоретично може статись, якщо клас в навчанні був відсутній),
                // то пропускаємо цей клас
                if class_count == 0.0 { 
                    continue;
                }
                let mut log_prob = (class_count / self.total_samples as f64).ln();  // натуральний логарифм
                // Для кожної ознаки додати log ймовірності P(feature=value | class=cls)
                for (i, value) in features.iter().enumerate() {
                    let count = self.feature_value_counts[i]
                                .get(cls).unwrap()  // карта для поточного класу (cls) 
                                .get(value).cloned().unwrap_or(0) as f64;
                    let class_total = *self.class_counts.get(cls).unwrap() as f64;
                    let distinct_values = self.feature_values[i].len() as f64;
                    // Ймовірність з згладжуванням:
                    let prob = (count + 1.0) / (class_total + distinct_values);
                    log_prob += prob.ln();
                }
                // Оновити найкращий клас, якщо поточний має більшу log-імовірність
                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class = cls.clone();
                }
            }
            predictions.push(best_class);
        }
        predictions
    }
}
