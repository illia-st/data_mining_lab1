use std::collections::HashMap;

pub struct OneRClassifier {
    best_feature: usize,
    rules: HashMap<String, String>,
    default_class: String,
}

impl OneRClassifier {
    pub fn new() -> Self {
        OneRClassifier {
            best_feature: 0,
            rules: HashMap::new(),
            default_class: String::new(),
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<String>>, y: &Vec<String>) {
        let num_features = x[0].len();
        let mut best_error = usize::MAX;
        let mut best_feature_index = 0;
        let mut best_rules: HashMap<String, String> = HashMap::new();

        let mut class_counts: HashMap<&String, usize> = HashMap::new();
        for label in y {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        if let Some((majority_class, _)) = class_counts.iter().max_by_key(|entry| entry.1) {
            self.default_class = (*majority_class).clone();
        }

        for fi in 0..num_features {
            let mut value_class_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
            for (index, (row, label)) in x.iter().zip(y.iter()).enumerate() {
                let value = &row[fi];
                value_class_counts
                    .entry(value.clone())
                    .or_insert(HashMap::new())
                    .entry(label.clone())
                    .or_insert(0_usize);
                *value_class_counts.get_mut(value).unwrap().get_mut(label).unwrap() += 1;
                let count = value_class_counts.get(value).unwrap().get(label).unwrap();
            }

            let mut rules: HashMap<String, String> = HashMap::new();
            let mut errors = 0;
            for (value, class_map) in &value_class_counts {
                let majority_class = class_map
                    .iter()
                    .max_by_key(|entry| entry.1)
                    .map(|(class, _)| class.clone())
                    .unwrap();
                rules.insert(value.clone(), majority_class.clone());

                let total_for_value: usize = class_map.values().sum();
                let correct_count = class_map.get(&majority_class).unwrap();
                errors += total_for_value - *correct_count;
            }

            if errors < best_error {
                best_error = errors;
                best_feature_index = fi;
                best_rules = rules;
            }
        }

        self.best_feature = best_feature_index;
        self.rules = best_rules;
    }

    pub fn predict(&self, data: &[Vec<String>]) -> Vec<String> {
        let mut predictions: Vec<String> = Vec::new();
        for features in data {
            if let Some(value) = features.get(self.best_feature) {
                let predicted = self.rules.get(value)
                                   .unwrap_or(&self.default_class)
                                   .clone();
                predictions.push(predicted);
            } else {
                predictions.push(self.default_class.clone());
            }
        }
        predictions
    }

    pub fn get_best_feature_index(&self) -> usize {
        self.best_feature
    }
}
