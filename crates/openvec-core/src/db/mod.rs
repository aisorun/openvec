/// OpenVec — Top-level Database API
///
/// This is the entry point developers interact with directly, providing a simple, SQLite-like embedded experience:
///
/// ```rust,no_run
/// # use openvec_core::{OpenVec, types::{Document, DistanceMetric, SearchRequest}};
/// # fn main() -> openvec_core::types::error::Result<()> {
/// let mut db = OpenVec::open("./data")?;
/// let mut coll = db.create_collection("articles", 768, DistanceMetric::Cosine)?;
/// coll.insert(Document::new("doc_1", vec![0.1_f32; 768]))?;
/// let results = coll.search(&SearchRequest::new(vec![0.1_f32; 768], 10))?;
/// # Ok(()) }
/// ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::collection::{Collection, CollectionConfig};
use crate::types::{DistanceMetric, Schema, VectorField};
use crate::types::error::{Error, Result};

pub mod hybrid;

// ─────────────────────────────────────────────
//  OpenVec Database Metadata (Persistent)
// ─────────────────────────────────────────────

/// Database-level persistent metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DbMeta {
    /// List of created collection configurations
    collections: HashMap<String, CollectionMeta>,
}

/// Metadata of a single collection (persistent portion)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionMeta {
    pub name: String,
    pub schema: Schema,
    pub created_at: i64,
}

// ─────────────────────────────────────────────
//  OpenVec Main Structure
// ─────────────────────────────────────────────

/// OpenVec database instance
///
/// Thread-safe (protects collection state via internal locking).
/// Supports embedded (in-process) usage, requiring no standalone server process.
pub struct OpenVec {
    data_dir: PathBuf,
    meta: parking_lot::RwLock<DbMeta>,
    /// Opened collections (loaded on demand)
    collections: parking_lot::RwLock<HashMap<String, Arc<Collection>>>,
    wal_sync: bool,
}

impl OpenVec {
    // ─────────────────────────────────────────────
    //  Initialization
    // ─────────────────────────────────────────────

    /// Opens (or creates) an OpenVec database
    ///
    /// # Parameters
    /// - `path`: Data directory (e.g. `./openvec_data` or `/var/lib/myapp/vectors`)
    ///
    /// # Behaviors
    /// - If directory does not exist, automatically creates it
    /// - Loads existing collection metadata
    /// - Does not open all collections immediately (lazy loading)
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let data_dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let meta = Self::load_meta(&data_dir).unwrap_or_default();

        info!("OpenVec opened at '{}', {} collections registered",
            data_dir.display(), meta.collections.len());

        Ok(Self {
            data_dir,
            meta: parking_lot::RwLock::new(meta),
            collections: parking_lot::RwLock::new(HashMap::new()),
            wal_sync: false,
        })
    }

    /// Set WAL sync policy (forces fsync on WAL append if true)
    pub fn with_wal_sync(mut self, wal_sync: bool) -> Self {
        self.wal_sync = wal_sync;
        self
    }

    // ─────────────────────────────────────────────
    //  Collection Management
    // ─────────────────────────────────────────────

    /// Creates a new collection
    ///
    /// # Parameters
    /// - `name`: Collection name (alphanumerics, underscores, hyphens)
    /// - `dimension`: Vector dimension
    /// - `metric`: Distance metric
    ///
    /// # Errors
    /// - Returns `Error::CollectionAlreadyExists` if a collection with the same name already exists
    pub fn create_collection(
        &self,
        name: impl Into<String>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<Arc<Collection>> {
        let name = name.into();
        Self::validate_collection_name(&name)?;

        {
            let meta = self.meta.read();
            if meta.collections.contains_key(&name) {
                return Err(Error::CollectionAlreadyExists(name));
            }
        }

        let schema = Schema::new().add_vector_field(
            VectorField::new("default", dimension).with_distance(metric)
        );

        let coll_meta = CollectionMeta {
            name: name.clone(),
            schema: schema.clone(),
            created_at: chrono::Utc::now().timestamp(),
        };

        {
            let mut meta = self.meta.write();
            meta.collections.insert(name.clone(), coll_meta);
            self.save_meta(&meta)?;
        }

        let config = CollectionConfig::new(name.clone(), schema).with_wal_sync(self.wal_sync);
        let coll = Arc::new(Collection::open(&self.data_dir, config)?);
        
        {
            let mut cols = self.collections.write();
            cols.insert(name.clone(), coll.clone());
        }

        Ok(coll)
    }

    /// Creates a collection using a custom Schema
    pub fn create_collection_with_schema(
        &self,
        name: impl Into<String>,
        schema: Schema,
    ) -> Result<Arc<Collection>> {
        let name = name.into();
        Self::validate_collection_name(&name)?;

        {
            let meta = self.meta.read();
            if meta.collections.contains_key(&name) {
                return Err(Error::CollectionAlreadyExists(name));
            }
        }

        let coll_meta = CollectionMeta {
            name: name.clone(),
            schema: schema.clone(),
            created_at: chrono::Utc::now().timestamp(),
        };

        {
            let mut meta = self.meta.write();
            meta.collections.insert(name.clone(), coll_meta);
            self.save_meta(&meta)?;
        }

        let config = CollectionConfig::new(name.clone(), schema).with_wal_sync(self.wal_sync);
        let coll = Arc::new(Collection::open(&self.data_dir, config)?);
        
        {
            let mut cols = self.collections.write();
            cols.insert(name.clone(), coll.clone());
        }

        Ok(coll)
    }

    /// Gets an existing collection (lazy loading)
    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>> {
        {
            let meta = self.meta.read();
            if !meta.collections.contains_key(name) {
                return Err(Error::CollectionNotFound(name.to_string()));
            }
        }

        // Lazy loading: open collection if not already in memory
        {
            let cols = self.collections.read();
            if let Some(coll) = cols.get(name) {
                return Ok(coll.clone());
            }
        }

        let mut cols = self.collections.write();
        // Double checked locking pattern
        if let Some(coll) = cols.get(name) {
            return Ok(coll.clone());
        }

        let coll_meta = {
            let meta = self.meta.read();
            meta.collections.get(name).cloned().unwrap()
        };

        let config = CollectionConfig::new(coll_meta.name.clone(), coll_meta.schema).with_wal_sync(self.wal_sync);
        let coll = Arc::new(Collection::open(&self.data_dir, config)?);
        cols.insert(name.to_string(), coll.clone());

        Ok(coll)
    }

    /// Gets an existing collection (read-only, assumes it is already loaded in memory)
    pub fn get_collection_read(&self, name: &str) -> Result<Arc<Collection>> {
        {
            let meta = self.meta.read();
            if !meta.collections.contains_key(name) {
                return Err(Error::CollectionNotFound(name.to_string()));
            }
        }
        
        let cols = self.collections.read();
        cols.get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))
    }

    /// Drops a collection (deletes all data files)
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        {
            let meta = self.meta.read();
            if !meta.collections.contains_key(name) {
                return Ok(false);
            }
        }

        // 关闭内存中的集合
        {
            let mut cols = self.collections.write();
            cols.remove(name);
        }

        // Delete collection data directory
        let coll_dir = self.data_dir.join(name);
        if coll_dir.exists() {
            std::fs::remove_dir_all(&coll_dir)?;
        }

        {
            let mut meta = self.meta.write();
            meta.collections.remove(name);
            self.save_meta(&meta)?;
        }

        info!("Collection '{}' dropped", name);
        Ok(true)
    }

    /// Checks if a collection exists
    pub fn collection_exists(&self, name: &str) -> bool {
        self.meta.read().collections.contains_key(name)
    }

    /// Lists all collection names
    pub fn list_collections(&self) -> Vec<String> {
        let meta = self.meta.read();
        let mut names: Vec<String> = meta.collections.keys().cloned().collect();
        names.sort();
        names
    }

    // ─────────────────────────────────────────────
    //  Internal Utilities
    // ─────────────────────────────────────────────

    fn meta_path(data_dir: &Path) -> PathBuf {
        data_dir.join("_meta.json")
    }

    fn load_meta(data_dir: &Path) -> Option<DbMeta> {
        let path = Self::meta_path(data_dir);
        if !path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn save_meta(&self, meta: &DbMeta) -> Result<()> {
        let path = Self::meta_path(&self.data_dir);
        let content = serde_json::to_string_pretty(meta)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        // Atomic write: write to temporary file first then rename
        let tmp_path = path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &content)?;
        std::fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    /// Validates collection name validity
    fn validate_collection_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(Error::InvalidCollectionName(
                name.to_string(),
                "name cannot be empty".to_string(),
            ));
        }
        if name.len() > 128 {
            return Err(Error::InvalidCollectionName(
                name.to_string(),
                "name too long (max 128 chars)".to_string(),
            ));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(Error::InvalidCollectionName(
                name.to_string(),
                "name must contain only alphanumerics, underscores, or hyphens".to_string(),
            ));
        }
        Ok(())
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::types::{Document, SearchRequest};

    #[test]
    fn db_create_and_search() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();

        let coll = db.create_collection("embeddings", 3, DistanceMetric::Cosine).unwrap();

        coll.insert(Document::new("a", vec![1.0, 0.0, 0.0])).unwrap();
        coll.insert(Document::new("b", vec![0.0, 1.0, 0.0])).unwrap();

        let req = SearchRequest::new(vec![1.0, 0.0, 0.0], 1);
        let results = coll.search(&req).unwrap();

        assert_eq!(results[0].id.as_str(), "a");
    }

    #[test]
    fn db_duplicate_collection_error() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();
        db.create_collection("test", 4, DistanceMetric::L2).unwrap();

        let result = db.create_collection("test", 4, DistanceMetric::L2);
        assert!(matches!(result, Err(Error::CollectionAlreadyExists(_))));
    }

    #[test]
    fn db_persistence() {
        let dir = tempdir().unwrap();

        // Create and write
        {
            let db = OpenVec::open(dir.path()).unwrap();
            let coll = db.create_collection("persist_test", 2, DistanceMetric::L2).unwrap();
            coll.insert(Document::new("p1", vec![1.0, 0.0])).unwrap();
        }

        // Reopen
        {
            let db = OpenVec::open(dir.path()).unwrap();
            assert!(db.collection_exists("persist_test"));
            let coll = db.get_collection("persist_test").unwrap();
            let doc = coll.get(&"p1".into()).unwrap();
            assert!(doc.is_some());
        }
    }

    #[test]
    fn db_drop_collection() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();

        db.create_collection("to_drop", 2, DistanceMetric::L2).unwrap();
        assert!(db.collection_exists("to_drop"));

        db.drop_collection("to_drop").unwrap();
        assert!(!db.collection_exists("to_drop"));
    }

    #[test]
    fn db_list_collections() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();

        db.create_collection("alpha", 2, DistanceMetric::L2).unwrap();
        db.create_collection("beta", 2, DistanceMetric::L2).unwrap();

        let names = db.list_collections();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn db_invalid_collection_name() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();

        let result = db.create_collection("invalid name!", 2, DistanceMetric::L2);
        assert!(matches!(result, Err(Error::InvalidCollectionName(_, _))));
    }
}
