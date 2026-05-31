/// Tokenizes text into lowercase alphanumeric terms
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|token| token.to_lowercase())
        .filter(|token| !token.is_empty() && token.len() > 1) // skip empty and single characters (like 'a', 'I')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let text = "Hello, world! Rust is AMAZING.";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "rust", "is", "amazing"]);
    }

    #[test]
    fn test_tokenize_empty_and_short() {
        let text = "a b c    !!!";
        let tokens = tokenize(text);
        assert!(tokens.is_empty());
    }
}
