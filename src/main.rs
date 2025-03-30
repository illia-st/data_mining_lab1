use std::error::Error;
use csv::Reader;

mod util;
mod one_r;
mod naive_bayes;
mod decision_tree;
mod knn;

fn one_r_test_call() -> Result<(), Box<dyn Error>> {
    let mut x: Vec<Vec<String>> = Vec::new();
    let mut y: Vec<String> = Vec::new();

    let mut reader = Reader::from_path("datasets/weather.csv")?;  // Відкрити файл (припускаємо наявність заголовку)
    let mut feature_count = 0;
    for record in reader.records() {
        let record = record?;
        feature_count = record.len();
        let class_value = record.get(feature_count - 1).unwrap().to_string();
        let features: Vec<String> = record.iter().take(feature_count - 1)
                                          .map(|s| s.to_string())
                                          .collect();
        x.push(features);
        y.push(class_value);
    }

    let feature_names: Vec<&str> = reader.headers()?.iter().take(feature_count - 1).collect();

    let mut model = one_r::OneRClassifier::new();
    model.fit(&x, &y);

    let best_feature = model.get_best_feature_index();

    println!("Best Feature is: {}", feature_names[best_feature]);

    let new_examples = vec![
        vec!["Sunny".to_string(), "Cool".to_string(), "Yes".to_string()],
        vec!["Rain".to_string(), "Mild".to_string(), "No".to_string()],
    ];
    let predictions = model.predict(&new_examples);
    println!("Predictions: {:?}", predictions);
    // Можливий результат: Predictions: ["No", "Yes"]

    Ok(())
}

fn naive_bayes_test_call() -> Result<(), Box<dyn Error>> { // need to finish with naive bayes
    // Зчитування даних із weather.csv
    let mut x: Vec<String> = Vec::new();
    let mut y: Vec<String> = Vec::new();
    // речення - предікт
    // let (mut x, mut y) = util::load_and_tokenize_dataset("datasets/spam.csv").unwrap();
    let mut reader = Reader::from_path("datasets/spam.csv")?;
    for record in reader.records() {
        let record = record?;
        let n = record.len();
        let class_value = record.get(0_usize).unwrap().to_string();
        let features = record.iter().skip(1_usize)
                                          .map(|s| s.to_string())
                                          .collect::<Vec<String>>()
                                          .join("");
        x.push(features);
        y.push(class_value);
    }

    // // Створення і навчання моделі наївного Байєса
    let mut nb_model = naive_bayes::NaiveBayesClassifier::new(1.);
    nb_model.fit(&x, &y);

    let test_message = "Congratulations! You've won a free iPhone. Click here to claim.";
    let res = nb_model.predict(test_message);
    println!("naive_bayes prediction: {res}");
    Ok(())
}

fn knn_test_call() -> Result<(), Box<dyn Error>> {
    let mut x_points: Vec<Vec<f64>> = Vec::new();
    let mut y_points: Vec<String> = Vec::new();
    let mut reader = Reader::from_path("datasets/iris.csv")?;
    for result in reader.records() {
        let record = result?;
        let num_fields = record.len();
    
        let class_val = record.get(num_fields - 1).unwrap().to_string();
    
        let features: Vec<f64> = record.iter()
            .skip(1) // skip the id
            .take_while(|val| val.parse::<f64>().is_ok())
            .map(|val| val.parse::<f64>().unwrap())
            .collect();
    
        x_points.push(features);
        y_points.push(class_val);
    }

    let mut knn_model = knn::KNNClassifier::new(3);
    knn_model.fit(&x_points, &y_points);

    // Тестові точки
    let test_points = vec![
        vec![5.4, 3.9, 1.7, 0.4],
        vec![5.8, 3.9, 1.7, 1.0],
    ];
    let predictions = knn_model.predict(&test_points);
    println!("kNN predictions: {:?}", predictions);

    Ok(())
}

fn decision_tree_call() -> Result<(), Box<dyn Error>> {
    let mut x: Vec<Vec<String>> = Vec::new();
    let mut y: Vec<String> = Vec::new();
    let mut reader = Reader::from_path("datasets/buy_computer.csv")?;
    let mut feature_count = 0;
    for record in reader.records() {
        let record = record?;
        feature_count = record.len();

        y.push(record.get(feature_count-1).unwrap().to_string());
        x.push(record.iter().take(feature_count-1).map(|s| s.to_string()).collect());
    }

    let feature_names: Vec<&str> = reader.headers()?.iter().take(feature_count - 1).collect();

    let mut tree_model = decision_tree::DecisionTreeClassifier::new();
    tree_model.fit(&x, &y);

    let new_examples = vec![
        vec!["31-40".into(), "High".into(), "Yes".into(), "Fair".into()],
    ];
    let predictions = tree_model.predict(&new_examples);
    println!("Decision Tree predictions: {:?}", predictions);

    tree_model.print_tree(feature_names.iter().as_slice(), "");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // one_r_test_call()
    knn_test_call()
    // decision_tree_call()
    // naive_bayes_test_call()
}
