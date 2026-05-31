/// Segment File (Immutable Persistent Storage)
///
/// Segment is the output of flushing a MemTable to disk; once written, it is never modified.
/// Zero-copy reads are achieved via memory-mapped I/O.
///
/// File Format:
/// ```text
/// [SegmentHeader: 256B]
/// [DocumentBlock] <- newline-delimited JSON serialized documents (JSONL)
/// [IndexBlock]    <- Reserved (serialized indexes in subsequent versions)
/// ```

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::types::{Document, DocumentId};
use crate::types::error::{Error, Result};

// ─────────────────────────────────────────────
//  Segment Header (fixed size)
// ─────────────────────────────────────────────

const SEGMENT_MAGIC: &[u8; 8] = b"OPENVEC\x01";

/// Segment file header (serialized to a fixed binary size)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentHeader {
    /// Magic number
    pub magic: [u8; 8],
    /// Format version
    pub version: u16,
    /// Segment ID (monotonically increasing)
    pub segment_id: u64,
    /// Document count
    pub doc_count: u32,
    /// Creation time (Unix timestamp, seconds)
    pub created_at: i64,
    /// DocumentBlock offset
    pub doc_block_offset: u64,
    /// DocumentBlock size (bytes)
    pub doc_block_size: u64,
}

impl SegmentHeader {
    fn new(segment_id: u64, doc_count: u32) -> Self {
        Self {
            magic: *SEGMENT_MAGIC,
            version: 1,
            segment_id,
            doc_count,
            created_at: chrono::Utc::now().timestamp(),
            doc_block_offset: 0,  // Filled during write
            doc_block_size: 0,
        }
    }

    /// Serializes to a fixed byte size (JSON + padding = 256B)
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(self).unwrap();
        // 填充到 256 字节（方便定位 doc_block_offset）
        bytes.resize(256, 0);
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Find JSON end position (first null byte)
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        serde_json::from_slice(&bytes[..end])
            .map_err(|e| Error::Deserialization(e.to_string()))
    }
}

const HEADER_PADDED_SIZE: usize = 256;

// ─────────────────────────────────────────────
//  Segment Writer
// ─────────────────────────────────────────────

/// Writes a batch of documents to a Segment file
pub struct SegmentWriter {
    path: PathBuf,
}

impl SegmentWriter {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf() }
    }

    /// Writes document list to Segment file
    pub fn write(&self, segment_id: u64, docs: &[&Document]) -> Result<SegmentMeta> {
        let doc_count = docs.len() as u32;

        // Serialize document block (JSONL format, newline delimited)
        let mut doc_bytes = Vec::new();
        for doc in docs {
            let line = serde_json::to_vec(doc)
                .map_err(|e| Error::Serialization(e.to_string()))?;
            doc_bytes.extend_from_slice(&line);
            doc_bytes.push(b'\n');
        }

        // Build header
        let mut header = SegmentHeader::new(segment_id, doc_count);
        header.doc_block_offset = HEADER_PADDED_SIZE as u64;
        header.doc_block_size = doc_bytes.len() as u64;

        // Write file
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);

        let header_bytes = header.to_bytes();
        assert_eq!(header_bytes.len(), HEADER_PADDED_SIZE);
        writer.write_all(&header_bytes)?;
        writer.write_all(&doc_bytes)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;

        Ok(SegmentMeta {
            segment_id,
            path: self.path.clone(),
            doc_count: doc_count as usize,
        })
    }
}

/// Segment metadata (held in memory)
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    pub segment_id: u64,
    pub path: PathBuf,
    pub doc_count: usize,
}

// ─────────────────────────────────────────────
//  Segment Reader (mmap)
// ─────────────────────────────────────────────

/// Reads Segment file via memory mapping (zero-copy)
pub struct SegmentReader {
    _file: File,
    mmap: Mmap,
    header: SegmentHeader,
}

impl SegmentReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_PADDED_SIZE {
            return Err(Error::Corruption("Segment file too small".to_string()));
        }

        let header = SegmentHeader::from_bytes(&mmap[..HEADER_PADDED_SIZE])?;

        // Validate magic
        if &header.magic != SEGMENT_MAGIC {
            return Err(Error::Corruption("Invalid segment magic".to_string()));
        }

        Ok(Self { _file: file, mmap, header })
    }

    /// Reads all documents
    pub fn read_all_documents(&self) -> Result<Vec<Document>> {
        let start = self.header.doc_block_offset as usize;
        let end = start + self.header.doc_block_size as usize;

        if end > self.mmap.len() {
            return Err(Error::Corruption("Document block out of bounds".to_string()));
        }

        let block = &self.mmap[start..end];
        let mut docs = Vec::new();

        for line in block.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let doc: Document = serde_json::from_slice(line)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            docs.push(doc);
        }

        Ok(docs)
    }

    /// Reads document ID set (fast scan)
    pub fn read_document_ids(&self) -> Result<Vec<DocumentId>> {
        let docs = self.read_all_documents()?;
        Ok(docs.into_iter().map(|d| d.id).collect())
    }

    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    pub fn doc_count(&self) -> usize {
        self.header.doc_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_doc(id: &str, vector: Vec<f32>) -> Document {
        Document::new(id, vector)
            .with_payload("title", id)
    }

    #[test]
    fn write_and_read_segment() {
        let dir = tempdir().unwrap();
        let seg_path = dir.path().join("seg_0001.seg");

        let docs = vec![
            make_doc("a", vec![1.0, 2.0]),
            make_doc("b", vec![3.0, 4.0]),
            make_doc("c", vec![5.0, 6.0]),
        ];
        let doc_refs: Vec<&Document> = docs.iter().collect();

        // Write
        let writer = SegmentWriter::new(&seg_path);
        let meta = writer.write(1, &doc_refs).unwrap();
        assert_eq!(meta.doc_count, 3);

        // Read
        let reader = SegmentReader::open(&seg_path).unwrap();
        assert_eq!(reader.doc_count(), 3);

        let read_docs = reader.read_all_documents().unwrap();
        assert_eq!(read_docs.len(), 3);
        assert_eq!(read_docs[0].id.as_str(), "a");
        assert_eq!(read_docs[2].id.as_str(), "c");
    }

    #[test]
    fn segment_header_roundtrip() {
        let h = SegmentHeader::new(42, 100);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_PADDED_SIZE);
        let h2 = SegmentHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h2.segment_id, 42);
        assert_eq!(h2.doc_count, 100);
    }
}
