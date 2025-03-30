use std::cmp::Ordering;

pub struct KNNClassifier {
    k: usize,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<String>,
}

impl KNNClassifier {
    pub fn new(k: usize) -> Self {
        KNNClassifier {
            k,
            x_train: Vec::new(),
            y_train: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<String>) {
        // Зберегти навчальні дані у структурі
        self.x_train = x.clone();
        self.y_train = y.clone();
    }

    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<String> {
        let mut predictions = Vec::new();
        for x in data {
            predictions.push(self.predict_one(x));
        }
        predictions
    }

    pub fn predict_one(&self, x: &Vec<f64>) -> String {
        let mut distances: Vec<(f64, &String)> = Vec::with_capacity(self.x_train.len());
        for (xi, yi) in self.x_train.iter().zip(self.y_train.iter()) {
            let mut dist_sq = 0.0;
            for (a, b) in xi.iter().zip(x.iter()) {
                let diff = a - b;
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();
            distances.push((dist, yi));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let k_nearest = &distances[..self.k.min(distances.len())];  // на випадок, якщо k > n

        let mut class_votes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for &(_, class) in k_nearest {
            *class_votes.entry(class.clone()).or_insert(0) += 1;
        }
        class_votes.into_iter().max_by_key(|entry| entry.1).unwrap().0
    }
}
