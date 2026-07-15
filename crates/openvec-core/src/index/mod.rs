/// Vector index module
///
/// Provides a unified `VectorIndex` trait along with concrete implementations:
/// - `FlatIndex`: Brute-force search, exact results, best for small datasets (< 10K vectors).
/// - `HnswIndex`: Hierarchical Navigable Small World graphs, high-performance ANN, best for medium scale (10K ~ 1M vectors).

pub mod flat;
pub mod hnsw;
pub mod ivf_sq8;

use crate::types::{DocumentId, SearchResult, VectorRef};
use crate::types::error::Result;

/// Unified interface that all vector indexes must implement
pub trait VectorIndex: Send + Sync {
    /// Inserts a vector
    fn insert(&mut self, id: &DocumentId, vector: VectorRef) -> Result<()>;

    /// Batch insert (default implementation calls insert one by one)
    fn batch_insert(&mut self, items: &[(&DocumentId, &[f32])]) -> Result<()> {
        for (id, vec) in items {
            self.insert(id, vec)?;
        }
        Ok(())
    }

    /// Soft delete: marks as deleted, skipped during searches
    fn delete(&mut self, id: &DocumentId) -> Result<bool>;

    /// Top-K nearest neighbors search (approximate or exact, depending on implementation)
    fn search(&self, query: VectorRef, k: usize, ef: Option<usize>, filter_fn: Option<&dyn Fn(&DocumentId) -> bool>) -> Result<Vec<SearchResult>>;

    /// Returns the number of active vectors in the index (excluding deleted ones)
    fn len(&self) -> usize;

    /// Whether the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index type name (used for diagnostics)
    fn index_type(&self) -> &'static str;

    /// Vector dimension
    fn dimension(&self) -> usize;

    /// Serializes the index to a byte vector
    fn serialize_to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserializes the index from a byte slice, replacing the current state
    fn deserialize_from_bytes(&mut self, bytes: &[u8]) -> Result<()>;
}
