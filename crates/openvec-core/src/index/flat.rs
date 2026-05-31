/// Flat Index (Brute-Force Search)
///
/// Performs exact Top-K search across all vectors.
/// - Advantages: Exact results (Recall = 1.0), simple implementation, updates take effect immediately.
/// - Disadvantages: O(n) search time, not suitable for large datasets.
/// - Applicability: Best for datasets with < 10K vectors, or scenarios requiring 100% exact results.

use std::collections::HashMap;
use parking_lot::RwLock;

use crate::distance::DistanceCalculator;
use crate::types::{DistanceMetric, DocumentId, SearchResult, VectorRef};
use crate::types::error::{Error, Result};
use super::VectorIndex;

/// Stores a single vector entry
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Entry {
    id: DocumentId,
    vector: Vec<f32>,
    deleted: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FlatIndexSerializable {
    entries: Vec<Entry>,
    dimension: usize,
}

/// Flat Index internal state
struct FlatIndexInner {
    entries: Vec<Entry>,
    id_to_pos: HashMap<String, usize>,
    dimension: usize,
    calc: DistanceCalculator,
    active_count: usize,
}

impl FlatIndexInner {
    fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            entries: Vec::new(),
            id_to_pos: HashMap::new(),
            dimension,
            calc: DistanceCalculator::new(metric),
            active_count: 0,
        }
    }

    fn insert(&mut self, id: &DocumentId, vector: VectorRef) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let mut vector_data = vector.to_vec();
        if self.calc.metric() == DistanceMetric::Cosine {
            crate::distance::normalize(&mut vector_data);
        }

        if let Some(&pos) = self.id_to_pos.get(id.as_str()) {
            // Update if already exists (restore if previously deleted)
            let entry = &mut self.entries[pos];
            if entry.deleted {
                entry.deleted = false;
                self.active_count += 1;
            }
            entry.vector.copy_from_slice(&vector_data);
            return Ok(());
        }

        let pos = self.entries.len();
        self.entries.push(Entry {
            id: id.clone(),
            vector: vector_data,
            deleted: false,
        });
        self.id_to_pos.insert(id.as_str().to_string(), pos);
        self.active_count += 1;
        Ok(())
    }

    fn delete(&mut self, id: &DocumentId) -> bool {
        if let Some(&pos) = self.id_to_pos.get(id.as_str()) {
            let entry = &mut self.entries[pos];
            if !entry.deleted {
                entry.deleted = true;
                self.active_count -= 1;
                return true;
            }
        }
        false
    }

    fn search(&self, query: VectorRef, k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        if self.active_count == 0 {
            return Ok(Vec::new());
        }

        let mut query_data = query.to_vec();
        if self.calc.metric() == DistanceMetric::Cosine {
            crate::distance::normalize(&mut query_data);
        }

        // Compute distances to all vectors, skipping deleted ones
        let mut distances: Vec<(f32, usize)> = self.entries.iter()
            .enumerate()
            .filter(|(_, e)| !e.deleted)
            .map(|(i, e)| {
                let dist = self.calc.compute(&query_data, &e.vector);
                (dist, i)
            })
            .collect();

        // Partial sort: only get the smallest k elements (O(n log k) instead of O(n log n))
        let actual_k = k.min(distances.len());
        distances.select_nth_unstable_by(actual_k.saturating_sub(1), |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances.truncate(actual_k);
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let results = distances.into_iter().map(|(score, idx)| {
            SearchResult {
                id: self.entries[idx].id.clone(),
                score,
                payload: None,
            }
        }).collect();

        Ok(results)
    }
}

/// Thread-safe Flat Index (using RwLock)
pub struct FlatIndex {
    inner: RwLock<FlatIndexInner>,
    dimension: usize,
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            inner: RwLock::new(FlatIndexInner::new(dimension, metric)),
            dimension,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&mut self, id: &DocumentId, vector: VectorRef) -> Result<()> {
        self.inner.write().insert(id, vector)
    }

    fn delete(&mut self, id: &DocumentId) -> Result<bool> {
        Ok(self.inner.write().delete(id))
    }

    fn search(&self, query: VectorRef, k: usize, _ef: Option<usize>) -> Result<Vec<SearchResult>> {
        self.inner.read().search(query, k)
    }

    fn len(&self) -> usize {
        self.inner.read().active_count
    }

    fn index_type(&self) -> &'static str {
        "flat"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let serializable = FlatIndexSerializable {
            entries: inner.entries.clone(),
            dimension: inner.dimension,
        };
        serde_json::to_vec(&serializable)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    fn deserialize_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let serializable: FlatIndexSerializable = serde_json::from_slice(bytes)
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        
        let mut inner = self.inner.write();
        inner.entries = serializable.entries;
        inner.dimension = serializable.dimension;
        
        // Rebuild id_to_pos and active_count safely without borrow clashes
        let mut id_to_pos = HashMap::new();
        let mut active_count = 0;
        for (i, entry) in inner.entries.iter().enumerate() {
            id_to_pos.insert(entry.id.to_string(), i);
            if !entry.deleted {
                active_count += 1;
            }
        }
        inner.id_to_pos = id_to_pos;
        inner.active_count = active_count;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;

    fn make_id(s: &str) -> DocumentId {
        DocumentId::from(s)
    }

    #[test]
    fn flat_insert_and_search_l2() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        idx.insert(&make_id("b"), &[1.0, 0.0]).unwrap();
        idx.insert(&make_id("c"), &[5.0, 0.0]).unwrap();

        let results = idx.search(&[0.1, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.as_str(), "a");
        assert_eq!(results[1].id.as_str(), "b");
    }

    #[test]
    fn flat_delete() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        idx.insert(&make_id("b"), &[0.1, 0.0]).unwrap();

        idx.delete(&make_id("a")).unwrap();
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id.as_str(), "b");
    }

    #[test]
    fn flat_dimension_mismatch() {
        let mut idx = FlatIndex::new(3, DistanceMetric::L2);
        let result = idx.insert(&make_id("x"), &[1.0, 2.0]); // wrong dimension
        assert!(matches!(result, Err(Error::DimensionMismatch { expected: 3, got: 2 })));
    }

    #[test]
    fn flat_upsert() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.insert(&make_id("a"), &[10.0, 0.0]).unwrap();
        // Update the same ID
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[0.0, 0.0], 1, None).unwrap();
        assert!((results[0].score).abs() < 1e-5);
    }

    #[test]
    fn flat_cosine_ordering() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Cosine);
        idx.insert(&make_id("same"), &[1.0, 0.0]).unwrap();
        idx.insert(&make_id("ortho"), &[0.0, 1.0]).unwrap();
        idx.insert(&make_id("opp"), &[-1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 3, None).unwrap();
        assert_eq!(results[0].id.as_str(), "same");
        assert_eq!(results[2].id.as_str(), "opp");
    }

    #[test]
    fn flat_k_larger_than_size() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        let results = idx.search(&[0.0, 0.0], 100, None).unwrap();
        assert_eq!(results.len(), 1);
    }
}
