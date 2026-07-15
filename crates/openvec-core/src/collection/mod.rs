/// Collection — Unified Management Layer for Vectors and Storage
///
/// Collection is the core concept of OpenVec, analogous to a "table" in relational databases.
/// Each Collection contains:
/// - One or more vector fields (each maintaining a separate vector index)
/// - Optional scalar fields (used for filtering)
/// - A StorageEngine (persisting all documents)
/// - A Full-text InvertedIndex (enabling lexical keyword search)

use std::collections::HashMap;
use std::path::Path;
use tracing::{info, debug};

use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use crate::index::ivf_sq8::IvfSq8Index;
use crate::index::VectorIndex;
use crate::storage::StorageEngine;
use crate::fulltext::InvertedIndex;
use crate::types::{
    Document, DocumentId, Schema, SearchRequest, SearchResult,
    VectorField, DistanceMetric, ScalarValue, ScalarFieldType,
};
use crate::types::error::{Error, Result};

// ─────────────────────────────────────────────
//  Collection Configuration
// ─────────────────────────────────────────────

/// Configuration for creating a Collection
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub name: String,
    pub schema: Schema,
    /// Threshold of vector count to automatically select the index type (Flat below this value, HNSW otherwise)
    pub auto_index_threshold: usize,
    /// Whether to compress vectors using IVF-SQ8 instead of HNSW graph
    pub prefer_sq8: bool,
    /// Whether to enforce fsync on WAL write operations
    pub wal_sync: bool,
    /// Custom training threshold for IVF-SQ8 index
    pub ivf_sq8_training_threshold: Option<usize>,
    /// Whether to compress documents on disk using LZ4
    pub compress: bool,
}

impl CollectionConfig {
    pub fn new(name: impl Into<String>, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
            auto_index_threshold: 10_000,
            prefer_sq8: false,
            wal_sync: false,
            ivf_sq8_training_threshold: None,
            compress: false,
        }
    }

    /// Quick-create configuration: Collection name + Dimension + Distance metric
    pub fn simple(name: impl Into<String>, dimension: usize, metric: DistanceMetric) -> Self {
        let schema = Schema::new().add_vector_field(
            VectorField::new("default", dimension).with_distance(metric)
        );
        Self::new(name, schema)
    }

    pub fn with_prefer_sq8(mut self) -> Self {
        self.prefer_sq8 = true;
        self
    }

    pub fn with_wal_sync(mut self, wal_sync: bool) -> Self {
        self.wal_sync = wal_sync;
        self
    }

    pub fn with_ivf_sq8_training_threshold(mut self, threshold: usize) -> Self {
        self.ivf_sq8_training_threshold = Some(threshold);
        self
    }

    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }
}

// ─────────────────────────────────────────────
//  Vector Index Pool (one index per vector field)
// ─────────────────────────────────────────────

/// Adaptive vector index: automatically switches between Flat and HNSW/IVF-SQ8 based on dataset size
enum AdaptiveIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
    IvfSq8(IvfSq8Index),
}

impl AdaptiveIndex {
    fn new_flat(dim: usize, metric: DistanceMetric) -> Self {
        Self::Flat(FlatIndex::new(dim, metric))
    }

    fn as_trait_mut(&mut self) -> &mut dyn VectorIndex {
        match self {
            Self::Flat(idx) => idx,
            Self::Hnsw(idx) => idx,
            Self::IvfSq8(idx) => idx,
        }
    }

    fn as_trait(&self) -> &dyn VectorIndex {
        match self {
            Self::Flat(idx) => idx,
            Self::Hnsw(idx) => idx,
            Self::IvfSq8(idx) => idx,
        }
    }

    fn index_type(&self) -> &'static str {
        match self {
            Self::Flat(_) => "flat",
            Self::Hnsw(_) => "hnsw",
            Self::IvfSq8(_) => "ivf_sq8",
        }
    }

    fn len(&self) -> usize {
        self.as_trait().len()
    }

    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let type_name = self.index_type();
        let index_bytes = match self {
            Self::Flat(idx) => idx.serialize_to_bytes()?,
            Self::Hnsw(idx) => idx.serialize_to_bytes()?,
            Self::IvfSq8(idx) => idx.serialize_to_bytes()?,
        };
        
        let serializable = (type_name.to_string(), index_bytes);
        serde_json::to_vec(&serializable)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        let (type_name, index_bytes): (String, Vec<u8>) = serde_json::from_slice(bytes)
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        
        match type_name.as_str() {
            "flat" => {
                let mut idx = FlatIndex::new(0, DistanceMetric::Cosine);
                idx.deserialize_from_bytes(&index_bytes)?;
                Ok(Self::Flat(idx))
            }
            "hnsw" => {
                let mut idx = HnswIndex::with_defaults(0, DistanceMetric::Cosine);
                idx.deserialize_from_bytes(&index_bytes)?;
                Ok(Self::Hnsw(idx))
            }
            "ivf_sq8" => {
                let mut idx = IvfSq8Index::new(0, DistanceMetric::Cosine);
                idx.deserialize_from_bytes(&index_bytes)?;
                Ok(Self::IvfSq8(idx))
            }
            _ => Err(Error::Corruption(format!("Unknown index type: {type_name}"))),
        }
    }
}

// ─────────────────────────────────────────────
//  Collection
// ─────────────────────────────────────────────

struct CollectionInner {
    storage: StorageEngine,
    /// Index for each vector field (field name → index)
    indexes: HashMap<String, AdaptiveIndex>,
    /// Full-text inverted index for lexical keyword queries
    fulltext_index: InvertedIndex,
}

impl CollectionInner {
    fn total_indexed_count(&self) -> usize {
        self.indexes.values().map(|idx| idx.len()).max().unwrap_or(0)
    }

    fn save_index_inner(&self) -> Result<()> {
        let mut serialized = HashMap::new();
        for (name, idx) in &self.indexes {
            let bytes = idx.serialize_to_bytes()?;
            serialized.insert(name.clone(), bytes);
        }

        // Serialize and save full-text inverted index as well
        if let Ok(ft_bytes) = self.fulltext_index.serialize_to_bytes() {
            serialized.insert("_fulltext".to_string(), ft_bytes);
        }

        let bytes = serde_json::to_vec(&serialized)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        let index_path = self.storage.data_dir().join("index_default.json");
        let tmp_path = index_path.with_extension("json.tmp");

        std::fs::write(&tmp_path, &bytes)?;
        std::fs::rename(&tmp_path, &index_path)?;

        debug!("Saved collection indexes to '{}'", index_path.display());
        Ok(())
    }

    fn rebuild_indexes(&mut self, config: &CollectionConfig) -> Result<()> {
        let docs = self.storage.all_documents()?;
        for doc in &docs {
            self.index_document(doc, &config.schema)?;
        }
        debug!("Rebuilt indexes from {} documents", docs.len());
        Ok(())
    }

    fn index_document(&mut self, doc: &Document, schema: &Schema) -> Result<()> {
        for (field_name, idx) in &mut self.indexes {
            if let Some(vector) = doc.vectors.get(field_name) {
                idx.as_trait_mut().insert(&doc.id, vector)?;
            }
        }

        // Index full-text scalar fields if present
        for sf in &schema.scalar_fields {
            if sf.field_type == ScalarFieldType::FullText {
                if let Some(ScalarValue::Text(val)) = doc.payload.get(&sf.name) {
                    self.fulltext_index.insert(&doc.id, val);
                }
            }
        }

        Ok(())
    }

    fn maybe_upgrade_indexes(&mut self, config: &CollectionConfig) -> Result<()> {
        let threshold = config.auto_index_threshold;
        let mut to_upgrade = Vec::new();

        for (field_name, idx) in &self.indexes {
            if matches!(idx, AdaptiveIndex::Flat(_)) && idx.len() >= threshold {
                to_upgrade.push(field_name.clone());
            }
        }

        for field_name in to_upgrade {
            if config.prefer_sq8 {
                info!("Auto-upgrading '{}' field index: Flat → IVF-SQ8", field_name);
                self.upgrade_to_ivf_sq8(&field_name, config)?;
            } else {
                info!("Auto-upgrading '{}' field index: Flat → HNSW", field_name);
                self.upgrade_to_hnsw(&field_name, config)?;
            }
        }

        Ok(())
    }

    fn upgrade_to_hnsw(&mut self, field_name: &str, config: &CollectionConfig) -> Result<()> {
        let vf = config.schema.get_vector_field(field_name)
            .ok_or_else(|| Error::VectorFieldNotFound(field_name.to_string()))?;

        let mut new_idx = HnswIndex::with_defaults(vf.dimension, vf.distance);

        // Rebuild HNSW index from storage
        let docs = self.storage.all_documents()?;
        for doc in &docs {
            if let Some(vector) = doc.vectors.get(field_name) {
                new_idx.insert(&doc.id, vector)?;
            }
        }

        self.indexes.insert(field_name.to_string(), AdaptiveIndex::Hnsw(new_idx));
        Ok(())
    }

    fn upgrade_to_ivf_sq8(&mut self, field_name: &str, config: &CollectionConfig) -> Result<()> {
        let vf = config.schema.get_vector_field(field_name)
            .ok_or_else(|| Error::VectorFieldNotFound(field_name.to_string()))?;

        let mut new_idx = IvfSq8Index::new(vf.dimension, vf.distance);
        if let Some(threshold) = config.ivf_sq8_training_threshold {
            new_idx.set_training_threshold(threshold);
        }

        // Rebuild IVF-SQ8 index from storage
        let docs = self.storage.all_documents()?;
        for doc in &docs {
            if let Some(vector) = doc.vectors.get(field_name) {
                new_idx.insert(&doc.id, vector)?;
            }
        }

        self.indexes.insert(field_name.to_string(), AdaptiveIndex::IvfSq8(new_idx));
        Ok(())
    }
}

/// Collection
pub struct Collection {
    config: CollectionConfig,
    inner: parking_lot::RwLock<CollectionInner>,
}

impl Collection {
    /// Creates or opens a collection
    pub fn open(data_dir: impl AsRef<Path>, config: CollectionConfig) -> Result<Self> {
        let name = config.name.clone();
        let storage = StorageEngine::open(data_dir.as_ref(), &name, config.wal_sync, config.compress)?;

        // Try to load index from disk first
        let mut indexes = HashMap::new();
        let mut loaded_indexes = false;
        let mut fulltext_index = InvertedIndex::new();

        let index_path = storage.data_dir().join("index_default.json");

        if index_path.exists() {
            if let Ok(bytes) = std::fs::read(&index_path) {
                if let Ok(serialized_indexes) = serde_json::from_slice::<HashMap<String, Vec<u8>>>(&bytes) {
                    let mut temp_indexes = HashMap::new();
                    let mut success = true;
                    for (field_name, index_bytes) in serialized_indexes {
                        if field_name == "_fulltext" {
                            let _ = fulltext_index.deserialize_from_bytes(&index_bytes);
                            continue;
                        }
                        if let Ok(idx) = AdaptiveIndex::deserialize_from_bytes(&index_bytes) {
                            temp_indexes.insert(field_name, idx);
                        } else {
                            success = false;
                            break;
                        }
                    }
                    if success && !temp_indexes.is_empty() {
                        indexes = temp_indexes;
                        loaded_indexes = true;
                    }
                }
            }
        }

        if !loaded_indexes {
            // Initialize index for each vector field
            for vf in &config.schema.vector_fields {
                let idx = AdaptiveIndex::new_flat(vf.dimension, vf.distance);
                indexes.insert(vf.name.clone(), idx);
            }
        }

        let mut inner = CollectionInner {
            storage,
            indexes,
            fulltext_index,
        };

        if !loaded_indexes {
            // Recover existing data from storage into index
            inner.rebuild_indexes(&config)?;
        } else {
            // Replay outstanding WAL / MemTable data to catch up!
            // First, re-index active documents in MemTable
            let memtable_docs = inner.storage.memtable_documents();
            for doc in &memtable_docs {
                inner.index_document(doc, &config.schema)?;
            }
            
            // Second, apply deletes from the MemTable
            let memtable_deleted = inner.storage.memtable_deleted();
            for del_id in &memtable_deleted {
                for idx in inner.indexes.values_mut() {
                    idx.as_trait_mut().delete(del_id)?;
                }
                inner.fulltext_index.delete(del_id);
            }
            
            // Rebuild fulltext index from all active documents if it wasn't loaded from disk
            let has_fulltext_fields = config.schema.scalar_fields.iter().any(|sf| sf.field_type == ScalarFieldType::FullText);
            if inner.fulltext_index.is_empty() && has_fulltext_fields {
                let all_docs = inner.storage.all_documents()?;
                for doc in &all_docs {
                    for sf in &config.schema.scalar_fields {
                        if sf.field_type == ScalarFieldType::FullText {
                            if let Some(ScalarValue::Text(val)) = doc.payload.get(&sf.name) {
                                inner.fulltext_index.insert(&doc.id, val);
                            }
                        }
                    }
                }
            }
        }

        let total_indexed = inner.total_indexed_count();
        let collection = Self {
            config,
            inner: parking_lot::RwLock::new(inner),
        };

        info!("Collection '{}' opened, {} docs indexed", name, total_indexed);
        Ok(collection)
    }

    /// Saves the in-memory indexes to a file
    pub fn save_index(&self) -> Result<()> {
        self.inner.read().save_index_inner()
    }

    /// Inserts or updates a document
    pub fn insert(&self, doc: Document) -> Result<DocumentId> {
        let id = doc.id.clone();
        let mut inner = self.inner.write();
        // 1. Update index first (if it fails, do not write to storage)
        inner.index_document(&doc, &self.config.schema)?;
        
        let before_flush_count = inner.storage.segment_count();
        // 2. Write to storage (WAL + MemTable)
        inner.storage.insert(doc)?;
        let after_flush_count = inner.storage.segment_count();
        
        // 3. Auto upgrade index type (Flat → HNSW / IVF-SQ8)
        inner.maybe_upgrade_indexes(&self.config)?;
        
        if after_flush_count != before_flush_count {
            inner.save_index_inner()?;
        }
        Ok(id)
    }

    /// Batch insert documents
    pub fn batch_insert(&self, docs: Vec<Document>) -> Result<Vec<DocumentId>> {
        let mut ids = Vec::with_capacity(docs.len());
        let mut inner = self.inner.write();
        let before_flush_count = inner.storage.segment_count();
        for doc in docs {
            ids.push(doc.id.clone());
            inner.index_document(&doc, &self.config.schema)?;
            inner.storage.insert(doc)?;
            inner.maybe_upgrade_indexes(&self.config)?;
        }
        let after_flush_count = inner.storage.segment_count();
        if after_flush_count != before_flush_count {
            inner.save_index_inner()?;
        }
        Ok(ids)
    }

    /// Deletes a document from vector indexes, BM25, and StorageEngine
    pub fn delete(&self, id: &DocumentId) -> Result<bool> {
        let mut inner = self.inner.write();
        // Delete from all vector indexes
        for idx in inner.indexes.values_mut() {
            idx.as_trait_mut().delete(id)?;
        }
        // Delete from full-text index
        inner.fulltext_index.delete(id);

        // Delete from storage
        inner.storage.delete(id)
    }

    /// Gets document by ID
    pub fn get(&self, id: &DocumentId) -> Result<Option<Document>> {
        self.inner.read().storage.get(id)
    }

    /// Vector search (with optional scalar filtering and RRF hybrid full-text fusion)
    pub fn search(&self, request: &SearchRequest) -> Result<Vec<SearchResult>> {
        let inner = self.inner.read();
        let vector_field = &request.vector_field;

        let idx = inner.indexes.get(vector_field)
            .ok_or_else(|| Error::VectorFieldNotFound(vector_field.clone()))?;

        // Determine candidate evaluation count
        let (k_search, filter) = if request.filter.is_some() {
            (request.limit * 10, request.filter.as_ref())
        } else {
            (request.limit, None)
        };

        // Create the pre-filter closure
        let check_match;
        let filter_fn = if let Some(ref f) = request.filter {
            let storage = &inner.storage;
            check_match = move |id: &DocumentId| -> bool {
                if let Ok(Some(doc)) = storage.get(id) {
                    f.matches(&doc)
                } else {
                    false
                }
            };
            Some(&check_match as &dyn Fn(&DocumentId) -> bool)
        } else {
            None
        };

        // 1. Perform retrieval (Hybrid RRF fusion or pure vector search)
        let mut results = if let Some(ref h_query) = request.hybrid_query {
            // Retrieve vector neighbors (over-sample slightly to ensure clean RRF ranks overlap)
            let vec_results = idx.as_trait().search(&request.vector, k_search.max(50), request.ef, filter_fn)?;
            // Retrieve BM25 text query matches
            let text_results = inner.fulltext_index.search(h_query, k_search.max(50));
            // Reciprocal Rank Fusion with weights
            let v_weight = request.vector_weight.unwrap_or(1.0);
            let t_weight = request.text_weight.unwrap_or(1.0);
            crate::db::hybrid::fuse_rrf_weighted(vec_results, text_results, v_weight, t_weight, k_search)
        } else {
            // Pure vector search
            idx.as_trait().search(&request.vector, k_search, request.ef, filter_fn)?
        };

        // 2. Apply metadata pre-filters if requested
        if let Some(f) = filter {
            let mut filtered = Vec::new();
            for result in results {
                if let Some(doc) = inner.storage.get(&result.id)? {
                    if f.matches(&doc) {
                        let mut r = result;
                        if request.with_payload {
                            r.payload = Some(doc.payload);
                        }
                        filtered.push(r);
                        if filtered.len() >= request.limit {
                            break;
                        }
                    }
                }
            }
            return Ok(filtered);
        }

        // 3. Truncate to request limit and optionally hydrate payloads
        results.truncate(request.limit);

        if request.with_payload {
            for result in &mut results {
                if let Some(doc) = inner.storage.get(&result.id)? {
                    result.payload = Some(doc.payload);
                }
            }
        }

        Ok(results)
    }

    /// Force flush memtable to segment (for tests/admin use)
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.storage.flush()?;
        Ok(())
    }

    /// Trigger background compaction (for tests/admin use)
    pub fn compact(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.storage.compact()?;
        Ok(())
    }

    /// Returns segment count (for tests/admin use)
    pub fn segment_count(&self) -> usize {
        self.inner.read().storage.segment_count()
    }

    // ─────────────────────────────────────────────
    //  Statistics & Diagnostics
    // ─────────────────────────────────────────────

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn doc_count(&self) -> usize {
        let inner = self.inner.read();
        inner.total_indexed_count()
    }

    pub fn index_types(&self) -> HashMap<String, &'static str> {
        let inner = self.inner.read();
        inner.indexes.iter().map(|(k, v)| (k.clone(), v.index_type())).collect()
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::types::{Document, ScalarField};

    fn make_doc(id: &str, v: Vec<f32>) -> Document {
        Document::new(id, v).with_payload("year", 2024i64)
    }

    #[test]
    fn collection_insert_and_search() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::simple("test", 3, DistanceMetric::L2);
        let coll = Collection::open(dir.path(), config).unwrap();

        coll.insert(make_doc("a", vec![0.0, 0.0, 0.0])).unwrap();
        coll.insert(make_doc("b", vec![1.0, 0.0, 0.0])).unwrap();
        coll.insert(make_doc("c", vec![5.0, 0.0, 0.0])).unwrap();

        let req = SearchRequest::new(vec![0.1, 0.0, 0.0], 2);
        let results = coll.search(&req).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.as_str(), "a");
    }

    #[test]
    fn collection_hybrid_search_rrf() {
        let dir = tempdir().unwrap();
        // Setup schema with full-text field
        let schema = Schema::new()
            .add_vector_field(VectorField::new("default", 2).with_distance(DistanceMetric::L2))
            .add_scalar_field(ScalarField::full_text("content"));

        let config = CollectionConfig::new("hybrid_test", schema);
        let coll = Collection::open(dir.path(), config).unwrap();

        // doc1: close vector, unrelated text
        coll.insert(Document::new("doc1", vec![0.0, 0.0])
            .with_payload("content", "Completely unrelated physics text.")).unwrap();
        // doc2: far vector, highly matching text
        coll.insert(Document::new("doc2", vec![10.0, 10.0])
            .with_payload("content", "Rust programming in vector databases.")).unwrap();

        // Search with hybrid query
        let req = SearchRequest::new(vec![0.1, 0.0], 2)
            .with_hybrid_query("Rust vector database");

        let results = coll.search(&req).unwrap();
        assert_eq!(results.len(), 2);

        // Hybrid rank merges both, bringing BOTH doc1 (vector match) and doc2 (fulltext match) into results!
        assert!(results.iter().any(|r| r.id.as_str() == "doc1"));
        assert!(results.iter().any(|r| r.id.as_str() == "doc2"));
    }

    #[test]
    fn collection_index_persistence_reopen() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::simple("persist_test", 3, DistanceMetric::L2);
        
        {
            let coll = Collection::open(dir.path(), config.clone()).unwrap();
            coll.insert(make_doc("a", vec![1.0, 0.0, 0.0])).unwrap();
            coll.insert(make_doc("b", vec![0.0, 1.0, 0.0])).unwrap();
            
            // Manually flush to write a segment and trigger index save
            coll.flush().unwrap();
            coll.save_index().unwrap();
            
            // Add an item to the MemTable (outstanding after index save)
            coll.insert(make_doc("c", vec![0.0, 0.0, 1.0])).unwrap();
            
            // Delete an item from the MemTable (outstanding delete)
            coll.delete(&"b".into()).unwrap();
        }

        // Reopen collection from the same path
        {
            let coll = Collection::open(dir.path(), config).unwrap();
            assert_eq!(coll.doc_count(), 2); // "a" and "c" are active, "b" is deleted
            
            let req = SearchRequest::new(vec![0.0, 0.0, 1.1], 2);
            let results = coll.search(&req).unwrap();
            assert_eq!(results[0].id.as_str(), "c");
            assert_eq!(results[1].id.as_str(), "a");
            assert!(!results.iter().any(|r| r.id.as_str() == "b"));
        }
    }

    #[test]
    fn collection_lsm_compaction() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::simple("compact_test", 2, DistanceMetric::L2);
        let coll = Collection::open(dir.path(), config).unwrap();

        // 1. Flush multiple segments manually
        coll.insert(make_doc("doc_1", vec![1.0, 0.0])).unwrap();
        coll.flush().unwrap();

        coll.insert(make_doc("doc_2", vec![0.0, 1.0])).unwrap();
        coll.flush().unwrap();

        coll.insert(make_doc("doc_3", vec![1.1, 1.1])).unwrap();
        coll.flush().unwrap();

        assert_eq!(coll.segment_count(), 3);

        // 2. Perform a delete (creates a tombstone)
        coll.delete(&"doc_2".into()).unwrap();

        // 3. Trigger compaction manually
        coll.compact().unwrap();

        // 4. Verify results
        assert_eq!(coll.segment_count(), 1); // Consolidated to 1 segment
        assert_eq!(coll.doc_count(), 2);             // "doc_1" and "doc_3" active
        
        let req = SearchRequest::new(vec![0.0, 1.0], 2);
        let results = coll.search(&req).unwrap();
        // doc_2 should be purged completely, doc_3 should be closest to [0, 1]
        assert_eq!(results[0].id.as_str(), "doc_3");
        assert!(!results.iter().any(|r| r.id.as_str() == "doc_2"));
    }

    #[test]
    fn test_collection_pre_filtering() {
        let dir = tempdir().unwrap();
        let schema = Schema::new()
            .add_vector_field(VectorField::new("default", 2).with_distance(DistanceMetric::L2))
            .add_scalar_field(ScalarField::int("year"))
            .add_scalar_field(ScalarField::text("category"));

        let config = CollectionConfig::new("pre_filter_test", schema);
        let coll = Collection::open(dir.path(), config).unwrap();

        // Insert docs with metadata
        coll.insert(Document::new("doc1", vec![0.0, 0.0])
            .with_payload("year", 2020i64)
            .with_payload("category", "A")).unwrap();
        coll.insert(Document::new("doc2", vec![1.0, 0.0])
            .with_payload("year", 2024i64)
            .with_payload("category", "A")).unwrap();
        coll.insert(Document::new("doc3", vec![2.0, 0.0])
            .with_payload("year", 2024i64)
            .with_payload("category", "B")).unwrap();

        // Search for closest to [0, 0] with filter: year == 2024 and category == "A"
        use crate::types::{Filter, FilterCondition};
        let cond_year = FilterCondition::eq("year", 2024i64);
        let cond_cat = FilterCondition::eq("category", "A");
        
        let filter = Filter::And(vec![
            Filter::Condition(cond_year),
            Filter::Condition(cond_cat),
        ]);

        let req = SearchRequest::new(vec![0.1, 0.0], 2).with_filter(filter);
        let results = coll.search(&req).unwrap();

        // Should return only "doc2" since it matches the filter and is closest
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id.as_str(), "doc2");
    }
}
