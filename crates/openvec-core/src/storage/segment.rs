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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocLocation {
    Offset(u64),
    Detailed {
        offset: u64,
        size: u32,
    }
}

impl DocLocation {
    pub fn offset(&self) -> u64 {
        match self {
            Self::Offset(o) => *o,
            Self::Detailed { offset, .. } => *offset,
        }
    }

    pub fn size(&self) -> Option<u32> {
        match self {
            Self::Offset(_) => None,
            Self::Detailed { size, .. } => Some(*size),
        }
    }
}

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
    /// IndexBlock offset
    #[serde(default)]
    pub index_block_offset: u64,
    /// IndexBlock size (bytes)
    #[serde(default)]
    pub index_block_size: u64,
    /// Compression type (0 = None, 1 = LZ4)
    #[serde(default)]
    pub compression_type: u8,
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
            index_block_offset: 0,
            index_block_size: 0,
            compression_type: 0,
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
    pub fn write(&self, segment_id: u64, docs: &[&Document], compress: bool) -> Result<SegmentMeta> {
        let doc_count = docs.len() as u32;

        // Serialize document block (JSONL format, newline delimited, optionally compressed)
        let mut doc_bytes = Vec::new();
        let mut offsets = std::collections::HashMap::new();

        for doc in docs {
            let line = serde_json::to_vec(doc)
                .map_err(|e| Error::Serialization(e.to_string()))?;

            let bytes_to_write = if compress {
                lz4_flex::compress_prepend_size(&line)
            } else {
                line
            };

            let offset_in_block = doc_bytes.len() as u64;
            let file_offset = (HEADER_PADDED_SIZE as u64) + offset_in_block;
            let size = bytes_to_write.len() as u32;

            offsets.insert(doc.id.to_string(), DocLocation::Detailed {
                offset: file_offset,
                size,
            });

            doc_bytes.extend_from_slice(&bytes_to_write);
            if !compress {
                doc_bytes.push(b'\n'); // Keep it human readable JSONL if not compressed
            }
        }

        // Serialize index block (doc_id -> DocLocation)
        let index_bytes = serde_json::to_vec(&offsets)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Build header
        let mut header = SegmentHeader::new(segment_id, doc_count);
        header.doc_block_offset = HEADER_PADDED_SIZE as u64;
        header.doc_block_size = doc_bytes.len() as u64;
        header.index_block_offset = header.doc_block_offset + header.doc_block_size;
        header.index_block_size = index_bytes.len() as u64;
        header.compression_type = if compress { 1 } else { 0 };

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
        writer.write_all(&index_bytes)?;
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
    offsets: std::collections::HashMap<String, DocLocation>,
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

        let mut offsets = std::collections::HashMap::new();

        // 1. Try to read index block if it exists
        if header.index_block_size > 0 {
            let start = header.index_block_offset as usize;
            let end = start + header.index_block_size as usize;
            if end <= mmap.len() {
                let index_slice = &mmap[start..end];
                if let Ok(map) = serde_json::from_slice::<std::collections::HashMap<String, DocLocation>>(index_slice) {
                    offsets = map;
                }
            }
        }

        // 2. Fallback to scanning if index block is missing or corrupt (backward compatibility)
        if offsets.is_empty() && header.doc_count > 0 {
            let start = header.doc_block_offset as usize;
            let end = start + header.doc_block_size as usize;
            if end <= mmap.len() {
                let block = &mmap[start..end];
                let mut current_offset = start;
                // Parse line by line to build index in memory
                let mut lines = block.split(|&b| b == b'\n');
                while let Some(line) = lines.next() {
                    let line_len = line.len();
                    if line_len > 0 {
                        if let Ok(doc) = serde_json::from_slice::<Document>(line) {
                            offsets.insert(doc.id.to_string(), DocLocation::Offset(current_offset as u64));
                        }
                    }
                    current_offset += line_len + 1; // +1 for the newline
                }
            }
        }

        Ok(Self {
            _file: file,
            mmap,
            header,
            offsets,
        })
    }

    /// Reads all documents
    pub fn read_all_documents(&self) -> Result<Vec<Document>> {
        let mut docs = Vec::new();
        for doc_id in self.offsets.keys() {
            if let Some(doc) = self.read_document(doc_id)? {
                docs.push(doc);
            }
        }
        docs.sort_by_key(|d| self.offsets.get(d.id.as_str()).map(|l| l.offset()).unwrap_or(0));
        Ok(docs)
    }

    /// Reads a single document by ID using the offsets index
    pub fn read_document(&self, doc_id: &str) -> Result<Option<Document>> {
        if let Some(&loc) = self.offsets.get(doc_id) {
            let start = loc.offset() as usize;
            if start >= self.mmap.len() {
                return Err(Error::Corruption("Document offset out of bounds".to_string()));
            }

            let bytes = match loc.size() {
                Some(size) => {
                    let end = start + size as usize;
                    if end > self.mmap.len() {
                        return Err(Error::Corruption("Document size out of bounds".to_string()));
                    }
                    self.mmap[start..end].to_vec()
                }
                None => {
                    // Backward compatibility: read until newline
                    let mmap_slice = &self.mmap[start..];
                    let end_idx = mmap_slice.iter().position(|&b| b == b'\n').unwrap_or(mmap_slice.len());
                    mmap_slice[..end_idx].to_vec()
                }
            };

            let decompressed = if self.header.compression_type == 1 {
                lz4_flex::decompress_size_prepended(&bytes)
                    .map_err(|e| Error::Deserialization(e.to_string()))?
            } else {
                bytes
            };

            let doc: Document = serde_json::from_slice(&decompressed)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    /// Reads document ID set (fast scan)
    pub fn read_document_ids(&self) -> Result<Vec<DocumentId>> {
        Ok(self.offsets.keys().map(|k| DocumentId::from(k.clone())).collect())
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
        let meta = writer.write(1, &doc_refs, false).unwrap();
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
    fn write_and_read_segment_compressed() {
        let dir = tempdir().unwrap();
        let seg_path = dir.path().join("seg_0002.seg");

        let docs = vec![
            make_doc("a", vec![1.0, 2.0]),
            make_doc("b", vec![3.0, 4.0]),
            make_doc("c", vec![5.0, 6.0]),
        ];
        let doc_refs: Vec<&Document> = docs.iter().collect();

        // Write
        let writer = SegmentWriter::new(&seg_path);
        let meta = writer.write(2, &doc_refs, true).unwrap();
        assert_eq!(meta.doc_count, 3);

        // Read
        let reader = SegmentReader::open(&seg_path).unwrap();
        assert_eq!(reader.doc_count(), 3);
        assert_eq!(reader.header.compression_type, 1);

        let read_docs = reader.read_all_documents().unwrap();
        assert_eq!(read_docs.len(), 3);
        assert_eq!(read_docs[0].id.as_str(), "a");
        assert_eq!(read_docs[2].id.as_str(), "c");

        // Verify point query
        let doc_b = reader.read_document("b").unwrap().unwrap();
        assert_eq!(doc_b.id.as_str(), "b");
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
