use std::collections::HashMap;
use crate::types::{DocumentId, SearchResult};

/// Combines semantic vector results and lexical full-text results using Reciprocal Rank Fusion (RRF)
///
/// Formula: RRF_score(d) = 1 / (60 + rank_vector(d)) + 1 / (60 + rank_text(d))
pub fn fuse_rrf(
    vector_results: Vec<SearchResult>,
    text_results: Vec<(String, f32)>,
    limit: usize,
) -> Vec<SearchResult> {
    fuse_rrf_weighted(vector_results, text_results, 1.0, 1.0, limit)
}

/// Combines semantic vector results and lexical full-text results using Weighted Reciprocal Rank Fusion (RRF)
///
/// Formula: RRF_score(d) = w_vector * (1 / (60 + rank_vector(d))) + w_text * (1 / (60 + rank_text(d)))
pub fn fuse_rrf_weighted(
    vector_results: Vec<SearchResult>,
    text_results: Vec<(String, f32)>,
    vector_weight: f32,
    text_weight: f32,
    limit: usize,
) -> Vec<SearchResult> {
    let mut rrf_scores = HashMap::new();

    // 1. Process vector search ranking
    for (i, res) in vector_results.iter().enumerate() {
        let rank = i + 1;
        let rrf = vector_weight * (1.0 / (60.0 + rank as f32));
        *rrf_scores.entry(res.id.to_string()).or_insert(0.0) += rrf;
    }

    // 2. Process text search ranking
    for (j, (doc_id, _)) in text_results.iter().enumerate() {
        let rank = j + 1;
        let rrf = text_weight * (1.0 / (60.0 + rank as f32));
        *rrf_scores.entry(doc_id.clone()).or_insert(0.0) += rrf;
    }

    // 3. Sort by RRF score descending
    let mut sorted: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(limit);

    // 4. Map to SearchResults
    sorted
        .into_iter()
        .map(|(id_str, score)| SearchResult {
            id: DocumentId::from(id_str),
            score,
            payload: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion_logic() {
        let vec_res = vec![
            SearchResult { id: DocumentId::from("doc_a"), score: 0.1, payload: None },
            SearchResult { id: DocumentId::from("doc_b"), score: 0.2, payload: None },
        ];
        let text_res = vec![
            ("doc_b".to_string(), 12.0),
            ("doc_c".to_string(), 8.0),
        ];

        let fused = fuse_rrf(vec_res, text_res, 3);
        assert_eq!(fused.len(), 3);

        // doc_b was rank 2 in vec and rank 1 in text, so it should rank highest!
        assert_eq!(fused[0].id.as_str(), "doc_b");
        // doc_a was rank 1 in vec
        assert_eq!(fused[1].id.as_str(), "doc_a");
        // doc_c was rank 2 in text
        assert_eq!(fused[2].id.as_str(), "doc_c");
    }

    #[test]
    fn test_weighted_rrf_fusion() {
        let vec_res = vec![
            SearchResult { id: DocumentId::from("semantic_match"), score: 0.1, payload: None },
        ];
        let text_res = vec![
            ("lexical_match".to_string(), 10.0),
        ];

        // 1. Prioritize semantic (vector) match (vector_weight = 10.0, text_weight = 1.0)
        let fused_semantic = fuse_rrf_weighted(vec_res.clone(), text_res.clone(), 10.0, 1.0, 2);
        assert_eq!(fused_semantic[0].id.as_str(), "semantic_match");

        // 2. Prioritize lexical (text) match (vector_weight = 1.0, text_weight = 10.0)
        let fused_lexical = fuse_rrf_weighted(vec_res, text_res, 1.0, 10.0, 2);
        assert_eq!(fused_lexical[0].id.as_str(), "lexical_match");
    }
}
