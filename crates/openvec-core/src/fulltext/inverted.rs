use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::DocumentId;

/// Inverted Index with BM25 (Okapi) scoring
#[derive(Serialize, Deserialize)]
pub struct InvertedIndex {
    // term -> list of (doc_id_str, term_frequency)
    postings: HashMap<String, Vec<(String, u32)>>,
    // doc_id_str -> length of doc in words
    doc_lengths: HashMap<String, usize>,
    // Total word length across all documents
    total_length: usize,
    // Total count of indexed documents
    num_docs: usize,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            num_docs: 0,
        }
    }

    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, crate::types::error::Error> {
        serde_json::to_vec(self)
            .map_err(|e| crate::types::error::Error::Serialization(e.to_string()))
    }

    pub fn deserialize_from_bytes(&mut self, bytes: &[u8]) -> Result<(), crate::types::error::Error> {
        let deserialized: Self = serde_json::from_slice(bytes)
            .map_err(|e| crate::types::error::Error::Deserialization(e.to_string()))?;
        *self = deserialized;
        Ok(())
    }

    /// Inserts (or updates) a document's full-text field
    pub fn insert(&mut self, doc_id: &DocumentId, text: &str) {
        // Drop old postings first (handles updates cleanly)
        self.delete(doc_id);

        let tokens = super::tokenizer::tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let doc_id_str = doc_id.to_string();
        self.doc_lengths.insert(doc_id_str.clone(), tokens.len());
        self.total_length += tokens.len();
        self.num_docs += 1;

        // Calculate term frequencies in this document
        let mut tfs = HashMap::new();
        for token in tokens {
            *tfs.entry(token).or_insert(0) += 1;
        }

        // Add to postings lists
        for (term, tf) in tfs {
            self.postings
                .entry(term)
                .or_insert_with(Vec::new)
                .push((doc_id_str.clone(), tf));
        }
    }

    /// Deletes a document from the inverted index
    pub fn delete(&mut self, doc_id: &DocumentId) {
        let doc_id_str = doc_id.to_string();
        if let Some(len) = self.doc_lengths.remove(&doc_id_str) {
            self.total_length = self.total_length.saturating_sub(len);
            self.num_docs = self.num_docs.saturating_sub(1);

            // Clean up matching postings
            for postings_list in self.postings.values_mut() {
                postings_list.retain(|(d, _)| d != &doc_id_str);
            }
        }
    }

    /// Searches the index using BM25 scoring, returning ranked list of (doc_id, score)
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        let query_tokens = super::tokenizer::tokenize(query);
        if query_tokens.is_empty() || self.num_docs == 0 {
            return Vec::new();
        }

        let avgdl = self.total_length as f32 / self.num_docs as f32;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let n_docs = self.num_docs as f32;

        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in query_tokens {
            if let Some(postings_list) = self.postings.get(&term) {
                let df = postings_list.len() as f32;
                // Lucene variant: ensures IDF is always positive
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

                for (doc_id, tf) in postings_list {
                    let doc_len = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                    let tf = *tf as f32;

                    let tf_numerator = tf * (k1 + 1.0);
                    let tf_denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl));

                    let term_score = idf * (tf_numerator / tf_denominator);
                    *scores.entry(doc_id.clone()).or_insert(0.0) += term_score;
                }
            }
        }

        // Sort by score descending
        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(limit);
        sorted
    }

    pub fn len(&self) -> usize {
        self.num_docs
    }

    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_search() {
        let mut idx = InvertedIndex::new();
        idx.insert(&DocumentId::from("doc1"), "The quick brown fox jumps over the lazy dog.");
        idx.insert(&DocumentId::from("doc2"), "Rust is a systems programming language focused on safety and speed.");
        idx.insert(&DocumentId::from("doc3"), "Fast and safe, Rust has exceptional performance.");

        // Querying safe and systems
        let results = idx.search("systems safety", 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc2");

        // Querying safe and speed
        let results = idx.search("safe rust", 3);
        assert_eq!(results.len(), 2);
        // doc3 has higher density of "safe rust" compared to doc2
        assert_eq!(results[0].0, "doc3");
        assert_eq!(results[1].0, "doc2");

        // Update doc3
        idx.insert(&DocumentId::from("doc3"), "Completely different text.");
        let results = idx.search("safe rust", 3);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc2");
    }
}
