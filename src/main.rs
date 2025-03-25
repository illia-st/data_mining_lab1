use std::error::Error;
use csv::Reader;

mod util;
mod one_r;
mod naive_bayes;
mod decision_tree;
mod knn;

fn one_r_test_call() -> Result<(), Box<dyn Error>> {
    // Вектор для ознак та вектор для класів
    let mut x: Vec<Vec<String>> = Vec::new();
    let mut y: Vec<String> = Vec::new();

    // Зчитування CSV-файлу
    let mut reader = Reader::from_path("datasets/weather.csv")?;  // Відкрити файл (припускаємо наявність заголовку)
    // Пропустити заголовки (reader.records() в csv crate ігнорує заголовок автоматично, якщо він є)
    for record in reader.records() {
        let record = record?;  // розпакувати Result
        // Останнє поле - це клас, інші поля - ознаки
        let num_fields = record.len();
        let class_value = record.get(num_fields - 1).unwrap().to_string();
        let features: Vec<String> = record.iter().take(num_fields - 1)
                                          .map(|s| s.to_string())
                                          .collect();
        x.push(features);
        y.push(class_value);
    }

    // Створити і навчити модель OneR
    let mut model = one_r::OneRClassifier::new();
    model.fit(&x, &y);

    // Нові приклади для прогнозу
    let new_examples = vec![
        vec!["Sunny".to_string(), "Cool".to_string(), "Yes".to_string()],
        vec!["Rain".to_string(), "Mild".to_string(), "No".to_string()],
    ];
    let predictions = model.predict(&new_examples);
    println!("Predictions: {:?}", predictions);
    // Можливий результат: Predictions: ["No", "Yes"]

    Ok(())
}

fn naive_bayes_test_call() -> Result<(), Box<dyn Error>> {
    // Зчитування даних із weather.csv
    let mut x: Vec<Vec<String>> = Vec::new();
    let mut y: Vec<String> = Vec::new();
    // речення - предікт
    let (mut x, mut y) = util::load_and_tokenize_dataset("datasets/spam.csv").unwrap();
    // let mut reader = Reader::from_path("weather.csv")?;
    // for (mail, predict) in tokens.into_iter().zip(predicts.into_iter()) {
    //     let n = mail.len();
    //     // let class_value = predict;
    //     let features: Vec<String> = record.iter().take(n-1)
    //                                       .map(|s| s.to_string())
    //                                       .collect();
    //     x.push(features);
    //     y.push(class_value);
    // }

    // // Створення і навчання моделі наївного Байєса
    let mut nb_model = naive_bayes::NaiveBayesClassifier::new();
    nb_model.fit(&x, &y);

    // // Нові приклади для прогнозу
    // let new_examples = vec![
    //     vec!["Sunny".into(), "Cool".into(), "Yes".into()],
    //     vec!["Overcast".into(), "Mild".into(), "No".into()],
    // ];
    // let predictions = nb_model.predict(&new_examples);
    // println!("Naive Bayes predictions: {:?}", predictions);
    // // Можливий результат: ["No", "Yes"]

    Ok(())
}

fn knn_test_call() -> Result<(), Box<dyn Error>> {
    let mut x_points: Vec<Vec<f64>> = Vec::new();
    let mut y_points: Vec<String> = Vec::new();
    let mut reader = Reader::from_path("datasets/iris.csv")?;
    for result in reader.records() {
        let record = result?;
        let num_fields = record.len();
    
        // The last field is the class label
        let class_val = record.get(num_fields - 1).unwrap().to_string();
    
        // All previous fields are features
        let features: Vec<f64> = record.iter()
            .skip(1) // skip the id
            .take_while(|val| val.parse::<f64>().is_ok())
            .map(|val| val.parse::<f64>().unwrap())
            .collect();
    
        x_points.push(features);
        y_points.push(class_val);
    }

    // Створення і "навчання" kNN
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
fn main() -> Result<(), Box<dyn Error>> {
    // one_r_test_call()
    knn_test_call()
}
