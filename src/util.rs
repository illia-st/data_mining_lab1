use csv::Reader;
use regex::Regex;

pub fn load_and_tokenize_dataset(path: &str) -> Result<(Vec<Vec<String>>, Vec<String>), Box<dyn std::error::Error>> {
    let mut x: Vec<Vec<String>> = Vec::new();
    let mut y: Vec<String> = Vec::new();

    let mut reader = Reader::from_path(path)?;
    let re = Regex::new(r"\b\w+\b")?; // слова без пунктуації

    for result in reader.records() {
        let record = result?;
        let class = record.get(0).unwrap().to_string();
        let message = record.get(1).unwrap();

        // Токенізація (перетворення повідомлення в слова)
        let tokens: Vec<String> = re.find_iter(message)
            .map(|mat| mat.as_str().to_lowercase())
            .collect();

        x.push(tokens);
        y.push(class);
    }

    Ok((x, y))
}