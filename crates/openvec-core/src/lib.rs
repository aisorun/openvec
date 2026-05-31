/// OpenVec Core — Core Library Entry Point
///
/// Module Structure:
/// - `types`: Public type definitions
/// - `distance`: Distance computations (with SIMD optimization)
/// - `index`: Vector index engines (Flat, HNSW)
/// - `storage`: Persistent storage (WAL, MemTable, Segment)
/// - `collection`: Collection management layer
/// - `db`: Top-level Database API

pub mod types;
pub mod distance;
pub mod index;
pub mod storage;
pub mod collection;
pub mod db;
pub mod fulltext;

// Re-export commonly used types at crate root
pub use types::{
    CollectionId, DocumentId, Vector, VectorRef,
    DistanceMetric, Document, Schema, VectorField, ScalarField,
    SearchResult, SearchResults, ScalarValue, Filter, FilterCondition,
};
pub use distance::DistanceCalculator;
pub use db::OpenVec;
pub use collection::Collection;
pub use types::error::{Error, Result};
