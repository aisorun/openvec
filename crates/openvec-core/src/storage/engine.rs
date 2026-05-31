/// Storage Engine — Coordinates WAL, MemTable, and Segment
///
/// Responsibilities:
/// 1. All write operations write to WAL first, then update in-memory state.
/// 2. Flush MemTable to Segment when exceeding the threshold.
/// 3. Recover in-memory state via WAL replay during startup.
/// 4. Provide document read, write, and delete interfaces.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use tracing::{info, debug};

use crate::types::{Document, DocumentId};
use crate::types::error::Result;
use super::wal::{WalRecord, WalReader, WalWriter};
use super::segment::{SegmentMeta, SegmentReader, SegmentWriter};

// MemTable Flush threshold (configurable, defaults to 4096 documents)
const DEFAULT_MEMTABLE_FLUSH_THRESHOLD: usize = 4096;

// ─────────────────────────────────────────────
//  MemTable (In-Memory Write Buffer)
// ─────────────────────────────────────────────

/// In-memory document buffer (BTreeMap guarantees ordering, useful for range scans)
struct MemTable {
    docs: BTreeMap<String, Document>,   // doc_id → Document
    deleted: HashSet<String>,            // Deleted doc_id
    size: usize,                         // Approximate memory usage (sum of vector dimensions)
}

impl MemTable {
    fn new() -> Self {
        Self {
            docs: BTreeMap::new(),
            deleted: HashSet::new(),
            size: 0,
        }
    }

    fn insert(&mut self, doc: Document) {
        // Calculate approximate size (number of vector elements)
        let vec_size: usize = doc.vectors.values().map(|v| v.len()).sum();
        if let Some(old) = self.docs.get(&doc.id.0) {
            let old_size: usize = old.vectors.values().map(|v| v.len()).sum();
            self.size = self.size.saturating_sub(old_size);
        }
        self.size += vec_size;
        self.deleted.remove(&doc.id.0);
        self.docs.insert(doc.id.0.clone(), doc);
    }

    fn delete(&mut self, id: &DocumentId) -> bool {
        if self.docs.remove(id.as_str()).is_some() {
            self.size = self.size.saturating_sub(100); // Approximate
            self.deleted.insert(id.0.clone());
            true
        } else if !self.deleted.contains(id.as_str()) {
            self.deleted.insert(id.0.clone());
            true
        } else {
            false
        }
    }

    fn get(&self, id: &DocumentId) -> Option<&Document> {
        if self.deleted.contains(id.as_str()) {
            return None;
        }
        self.docs.get(id.as_str())
    }

    fn all_documents(&self) -> impl Iterator<Item = &Document> {
        self.docs.values().filter(|d| !self.deleted.contains(&d.id.0))
    }

    fn doc_count(&self) -> usize {
        self.docs.len()
    }

    fn is_full(&self, threshold: usize) -> bool {
        self.doc_count() >= threshold
    }

    fn clear(&mut self) {
        self.docs.clear();
        self.deleted.clear();
        self.size = 0;
    }
}

// ─────────────────────────────────────────────
//  Storage Engine
// ─────────────────────────────────────────────

/// Storage Engine for a single Collection
pub struct StorageEngine {
    collection_name: String,
    data_dir: PathBuf,
    wal: WalWriter,
    memtable: MemTable,
    segments: Vec<SegmentMeta>,
    next_segment_id: u64,
    flush_threshold: usize,
    /// Set of deleted doc_ids across all Segments (maintained in memory)
    segment_deleted: HashSet<String>,
    wal_sync: bool,
}

impl StorageEngine {
    /// Opens collection storage (automatically creates directories and recovers from WAL)
    pub fn open(data_dir: impl AsRef<Path>, collection_name: impl Into<String>, wal_sync: bool) -> Result<Self> {
        let collection_name = collection_name.into();
        let data_dir = data_dir.as_ref().to_path_buf();

        // Create collection data directory
        let coll_dir = data_dir.join(&collection_name);
        std::fs::create_dir_all(&coll_dir)?;

        let wal_path = coll_dir.join("current.wal");

        // Scan existing Segment files
        let mut segments = Self::scan_segments(&coll_dir)?;
        segments.sort_by_key(|s| s.segment_id);

        let next_segment_id = segments.last().map(|s| s.segment_id + 1).unwrap_or(0);

        // Open WAL writer
        let wal_writer = WalWriter::open(&wal_path, wal_sync)?;

        // Replay WAL
        let wal_reader = WalReader::new(&wal_path);
        let records = wal_reader.replay()?;

        let mut engine = Self {
            collection_name,
            data_dir: coll_dir,
            wal: wal_writer,
            memtable: MemTable::new(),
            segments,
            next_segment_id,
            flush_threshold: DEFAULT_MEMTABLE_FLUSH_THRESHOLD,
            segment_deleted: HashSet::new(),
            wal_sync,
        };

        // Apply WAL records (does not rewrite WAL, directly applies to MemTable)
        for record in records {
            engine.apply_wal_record(record)?;
        }

        info!("StorageEngine opened: collection='{}', segments={}, memtable_docs={}",
            engine.collection_name, engine.segments.len(), engine.memtable.doc_count());

        Ok(engine)
    }

    /// Scans all Segment files in the directory
    fn scan_segments(dir: &Path) -> Result<Vec<SegmentMeta>> {
        let mut segments = Vec::new();
        if !dir.exists() {
            return Ok(segments);
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("seg") {
                if let Ok(reader) = SegmentReader::open(&path) {
                    segments.push(SegmentMeta {
                        segment_id: reader.header().segment_id,
                        path,
                        doc_count: reader.doc_count(),
                    });
                }
            }
        }
        Ok(segments)
    }

    /// Applies a single WAL record to the in-memory state
    fn apply_wal_record(&mut self, record: WalRecord) -> Result<()> {
        match record {
            WalRecord::Insert { document, .. } => {
                self.memtable.insert(document);
            }
            WalRecord::Delete { doc_id, .. } => {
                self.memtable.delete(&doc_id);
            }
            WalRecord::Flush { segment_id, .. } => {
                // Flush record already exists in WAL, indicating this segment is persistent
                // MemTable was cleared during Flush; no action needed
                debug!("Replayed Flush for segment_id={segment_id}");
            }
        }
        Ok(())
    }

    // ─────────────────────────────────────────────
    //  Write Operations
    // ─────────────────────────────────────────────

    /// Inserts or updates a document
    pub fn insert(&mut self, doc: Document) -> Result<()> {
        // 1. Write to WAL first
        let record = WalRecord::Insert {
            collection: self.collection_name.clone(),
            document: doc.clone(),
        };
        self.wal.append(&record)?;

        // 2. Write to MemTable
        self.memtable.insert(doc);

        // 3. Flush if MemTable is full
        if self.memtable.is_full(self.flush_threshold) {
            self.flush()?;
            
            // Trigger LSM compaction if we have 4 or more segments
            if self.segment_count() >= 4 {
                self.compact()?;
            }
        }

        Ok(())
    }

    /// Batch insert documents
    pub fn batch_insert(&mut self, docs: Vec<Document>) -> Result<()> {
        for doc in docs {
            self.insert(doc)?;
        }
        Ok(())
    }

    /// Deletes a document
    pub fn delete(&mut self, id: &DocumentId) -> Result<bool> {
        // Write WAL
        let record = WalRecord::Delete {
            collection: self.collection_name.clone(),
            doc_id: id.clone(),
        };
        self.wal.append(&record)?;

        // Update memory
        let deleted_from_memtable = self.memtable.delete(id);
        // Mark segment deleted set regardless of whether it's in MemTable
        self.segment_deleted.insert(id.0.clone());

        Ok(deleted_from_memtable || self.segment_deleted.contains(id.as_str()))
    }

    // ─────────────────────────────────────────────
    //  Read Operations
    // ─────────────────────────────────────────────

    /// Gets document by ID (checks MemTable first, then Segments)
    pub fn get(&self, id: &DocumentId) -> Result<Option<Document>> {
        // Check if deleted
        if self.segment_deleted.contains(id.as_str()) {
            return Ok(None);
        }

        // Check MemTable first
        if let Some(doc) = self.memtable.get(id) {
            return Ok(Some(doc.clone()));
        }

        // Check Segments (starting from the newest Segment)
        for seg_meta in self.segments.iter().rev() {
            let reader = SegmentReader::open(&seg_meta.path)?;
            for doc in reader.read_all_documents()? {
                if doc.id == *id {
                    if !self.segment_deleted.contains(id.as_str()) {
                        return Ok(Some(doc));
                    }
                    return Ok(None);
                }
            }
        }

        Ok(None)
    }

    /// Returns all active documents (MemTable + Segments, deduplicated and filtered)
    pub fn all_documents(&self) -> Result<Vec<Document>> {
        let mut seen = HashSet::new();
        let mut docs = Vec::new();

        // Collect documents in MemTable first
        for doc in self.memtable.all_documents() {
            seen.insert(doc.id.0.clone());
            docs.push(doc.clone());
        }

        // Collect from Segments (older first, newer overrides)
        for seg_meta in self.segments.iter() {
            let reader = SegmentReader::open(&seg_meta.path)?;
            for doc in reader.read_all_documents()? {
                if !seen.contains(&doc.id.0) && !self.segment_deleted.contains(&doc.id.0) {
                    seen.insert(doc.id.0.clone());
                    docs.push(doc);
                }
            }
        }

        Ok(docs)
    }

    // ─────────────────────────────────────────────
    //  Flush (MemTable → Segment)
    // ─────────────────────────────────────────────

    /// Flushes documents in MemTable to a Segment file
    pub fn flush(&mut self) -> Result<Option<SegmentMeta>> {
        if self.memtable.doc_count() == 0 {
            return Ok(None);
        }

        let segment_id = self.next_segment_id;
        let seg_filename = format!("{:08}.seg", segment_id);
        let seg_path = self.data_dir.join(seg_filename);

        // Collect all documents in MemTable
        let docs: Vec<Document> = self.memtable.all_documents().cloned().collect();
        let doc_refs: Vec<&Document> = docs.iter().collect();

        // Write Segment
        let writer = SegmentWriter::new(&seg_path);
        let meta = writer.write(segment_id, &doc_refs)?;

        // Write WAL Flush record
        self.wal.append(&WalRecord::Flush {
            collection: self.collection_name.clone(),
            segment_id,
        })?;

        // Clear MemTable
        self.memtable.clear();
        self.next_segment_id += 1;
        self.segments.push(meta.clone());

        info!("Flushed segment_id={} with {} docs", segment_id, docs.len());
        Ok(Some(meta))
    }

    /// Consolidates all segment files into a single segment to purge tombstones and keep segment count low
    pub fn compact(&mut self) -> Result<()> {
        if self.segments.len() < 2 {
            return Ok(());
        }

        info!("Starting LSM compaction for collection '{}' (merging {} segments)",
            self.collection_name, self.segments.len());

        // 1. Gather all active documents across segments and MemTable
        let active_docs = self.all_documents()?;
        let doc_refs: Vec<&Document> = active_docs.iter().collect();

        // 2. Build the new consolidated segment
        let segment_id = self.next_segment_id;
        let seg_filename = format!("{:08}.seg", segment_id);
        let seg_path = self.data_dir.join(seg_filename);

        let writer = SegmentWriter::new(&seg_path);
        let consolidated_meta = writer.write(segment_id, &doc_refs)?;

        // 3. Remove old segment files
        let old_segments = std::mem::take(&mut self.segments);
        for old_seg in old_segments {
            if old_seg.path.exists() {
                let _ = std::fs::remove_file(&old_seg.path);
            }
        }

        // 4. Update memory structures
        self.segments = vec![consolidated_meta];
        self.next_segment_id += 1;
        self.segment_deleted.clear();
        self.memtable.clear();

        // 5. Truncate WAL safely by recreating it empty
        let wal_path = self.data_dir.join("current.wal");
        let _ = std::fs::remove_file(&wal_path);
        self.wal = WalWriter::open(&wal_path, self.wal_sync)?;

        info!("LSM compaction complete for collection '{}'. Consolidated into segment_id={}",
            self.collection_name, segment_id);

        Ok(())
    }

    // ─────────────────────────────────────────────
    //  Statistics
    // ─────────────────────────────────────────────

    pub fn memtable_documents(&self) -> Vec<Document> {
        self.memtable.all_documents().cloned().collect()
    }

    pub fn memtable_deleted(&self) -> Vec<DocumentId> {
        self.memtable.deleted.iter().map(|id| DocumentId::from(id.clone())).collect()
    }

    pub fn memtable_doc_count(&self) -> usize {
        self.memtable.doc_count()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn total_doc_count(&self) -> usize {
        let seg_count: usize = self.segments.iter().map(|s| s.doc_count).sum();
        self.memtable.doc_count() + seg_count
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::types::Document;

    fn make_doc(id: &str) -> Document {
        Document::new(id, vec![1.0, 2.0, 3.0])
            .with_payload("name", id)
    }

    #[test]
    fn engine_insert_and_get() {
        let dir = tempdir().unwrap();
        let mut engine = StorageEngine::open(dir.path(), "test_coll", true).unwrap();

        engine.insert(make_doc("doc1")).unwrap();
        engine.insert(make_doc("doc2")).unwrap();

        let d1 = engine.get(&DocumentId::from("doc1")).unwrap();
        assert!(d1.is_some());
        assert_eq!(d1.unwrap().id.as_str(), "doc1");

        let missing = engine.get(&DocumentId::from("doc_missing")).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn engine_delete() {
        let dir = tempdir().unwrap();
        let mut engine = StorageEngine::open(dir.path(), "test_coll", true).unwrap();

        engine.insert(make_doc("doc1")).unwrap();
        engine.delete(&DocumentId::from("doc1")).unwrap();

        let d = engine.get(&DocumentId::from("doc1")).unwrap();
        assert!(d.is_none());
    }

    #[test]
    fn engine_flush_and_read() {
        let dir = tempdir().unwrap();
        let mut engine = StorageEngine::open(dir.path(), "test_coll", true).unwrap();

        for i in 0..5 {
            engine.insert(make_doc(&format!("doc{i}"))).unwrap();
        }

        // Force Flush
        engine.flush().unwrap();
        assert_eq!(engine.memtable_doc_count(), 0);
        assert_eq!(engine.segment_count(), 1);

        // Read from Segment
        let docs = engine.all_documents().unwrap();
        assert_eq!(docs.len(), 5);
    }

    #[test]
    fn engine_wal_recovery() {
        let dir = tempdir().unwrap();

        // Write data
        {
            let mut engine = StorageEngine::open(dir.path(), "test_coll", true).unwrap();
            engine.insert(make_doc("doc_recover")).unwrap();
            // No flush, simulate crash
        }

        // Reopen and recover from WAL
        {
            let engine = StorageEngine::open(dir.path(), "test_coll", true).unwrap();
            let doc = engine.get(&DocumentId::from("doc_recover")).unwrap();
            assert!(doc.is_some(), "WAL recovery should restore doc_recover");
        }
    }
}
