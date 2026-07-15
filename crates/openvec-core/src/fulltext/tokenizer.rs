/// Tokenizes text into lowercase terms, with support for CJK N-Grams
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let raw_tokens = text.split(|c: char| !c.is_alphanumeric());

    for raw_token in raw_tokens {
        if raw_token.is_empty() {
            continue;
        }

        let chars: Vec<char> = raw_token.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if is_cjk_char(chars[i]) {
                // Collect contiguous CJK characters
                let mut cjk_buf = Vec::new();
                while i < chars.len() && is_cjk_char(chars[i]) {
                    cjk_buf.push(chars[i]);
                    i += 1;
                }

                // Generate Uni-Grams and Bi-Grams for CJK buffer
                for j in 0..cjk_buf.len() {
                    // Uni-Gram
                    tokens.push(cjk_buf[j].to_string());

                    // Bi-Gram
                    if j + 1 < cjk_buf.len() {
                        let mut bigram = String::new();
                        bigram.push(cjk_buf[j]);
                        bigram.push(cjk_buf[j + 1]);
                        tokens.push(bigram);
                    }
                }
            } else {
                // Collect contiguous non-CJK characters
                let mut non_cjk_buf = String::new();
                while i < chars.len() && !is_cjk_char(chars[i]) {
                    non_cjk_buf.push(chars[i]);
                    i += 1;
                }
                let lower = non_cjk_buf.to_lowercase();
                if !lower.is_empty() && lower.len() > 1 {
                    tokens.push(lower);
                }
            }
        }
    }

    tokens
}

fn is_cjk_char(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FFF).contains(&u)
        || (0x3400..=0x4DBF).contains(&u)
        || (0x20000..=0x2A6DF).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
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

    #[test]
    fn test_tokenize_cjk() {
        let text = "我是Rust数据库";
        let tokens = tokenize(text);
        assert!(tokens.contains(&"我".to_string()));
        assert!(tokens.contains(&"我是".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"数".to_string()));
        assert!(tokens.contains(&"数据".to_string()));
        assert!(tokens.contains(&"据库".to_string()));
    }
}
