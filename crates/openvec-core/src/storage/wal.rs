/// Write-Ahead Log (WAL)
///
/// WAL guarantees crash safety: before modifying the memory state, the operation is written to the log file.
/// On restart, the un-flushed memory state is restored by replaying the WAL.
///
/// Log Entry Format (binary):
/// ```text
/// [magic: 4B][crc32: 4B][length: 4B][payload: length B]
/// ```
/// magic = 0x4F564543 ("OVEC")
///
/// Payload is the JSON serialized WalRecord.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::types::{Document, DocumentId};
use crate::types::error::{Error, Result};

// ─────────────────────────────────────────────
//  WAL Record Types
// ─────────────────────────────────────────────

/// Operation types recorded in the WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalRecord {
    /// Insert or update a document
    Insert {
        collection: String,
        document: Document,
    },
    /// Delete a document
    Delete {
        collection: String,
        doc_id: DocumentId,
    },
    /// Segment flushed to disk, preceding WAL can be truncated
    Flush {
        collection: String,
        segment_id: u64,
    },
}

// ─────────────────────────────────────────────
//  WAL Constants
// ─────────────────────────────────────────────

const WAL_MAGIC: u32 = 0x4F564543; // "OVEC"
const HEADER_SIZE: usize = 12;     // magic(4) + crc32(4) + length(4)

// ─────────────────────────────────────────────
//  WAL Writer
// ─────────────────────────────────────────────

/// WAL Writer (append-only)
pub struct WalWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    entries_written: u64,
    sync_on_write: bool,
}

impl WalWriter {
    /// Opens (or creates) a WAL file
    pub fn open(path: impl AsRef<Path>, sync_on_write: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let entries_written = 0;
        info!("WAL opened: {}", path.display());
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entries_written,
            sync_on_write,
        })
    }

    /// Appends a WAL record (serializes + CRC + write + fsync if configured)
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let payload = serde_json::to_vec(record)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        let crc = crc32fast::hash(&payload);
        let length = payload.len() as u32;

        // Write header
        self.writer.write_all(&WAL_MAGIC.to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&length.to_le_bytes())?;
        // Write payload
        self.writer.write_all(&payload)?;

        // Flush (Important: ensures kernel buffer is written)
        self.writer.flush()?;
        
        // fsync guarantees durability if configured
        if self.sync_on_write {
            self.writer.get_ref().sync_data()?;
        }

        self.entries_written += 1;
        debug!("WAL append #{}: {:?}", self.entries_written, record);
        Ok(())
    }

    /// Returns the number of written records
    pub fn entries_written(&self) -> u64 {
        self.entries_written
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ─────────────────────────────────────────────
//  WAL Reader (for replaying)
// ─────────────────────────────────────────────

/// WAL Reader (for crash recovery)
pub struct WalReader {
    path: PathBuf,
}

impl WalReader {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf() }
    }

    /// Replays all WAL records
    ///
    /// Stops replay if a corrupted record is encountered (logs warning without error).
    /// Returns all successfully parsed records.
    pub fn replay(&self) -> Result<Vec<WalRecord>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();

        loop {
            // Read header
            let mut header = [0u8; HEADER_SIZE];
            match reader.read_exact(&mut header) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let crc_stored = u32::from_le_bytes(header[4..8].try_into().unwrap());
            let length = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

            // Validate magic
            if magic != WAL_MAGIC {
                warn!("WAL corruption: invalid magic at offset, stopping replay");
                break;
            }

            // Read payload
            let mut payload = vec![0u8; length];
            match reader.read_exact(&mut payload) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    warn!("WAL truncated payload, stopping replay");
                    break;
                }
                Err(e) => return Err(e.into()),
            }

            // Validate CRC
            let crc_actual = crc32fast::hash(&payload);
            if crc_stored != crc_actual {
                warn!("WAL corruption: CRC mismatch (expected {crc_stored}, got {crc_actual}), stopping replay");
                break;
            }

            // Deserialize
            match serde_json::from_slice::<WalRecord>(&payload) {
                Ok(record) => {
                    debug!("WAL replay: {:?}", record);
                    records.push(record);
                }
                Err(e) => {
                    warn!("WAL deserialization failed: {e}, skipping");
                }
            }
        }

        info!("WAL replay complete: {} records", records.len());
        Ok(records)
    }
}

// ─────────────────────────────────────────────
//  CRC32 helper (internal implementation)
// ─────────────────────────────────────────────

mod crc32fast {
    /// Simple CRC32 (Castagnoli) computation
    /// Uses a lookup table; zero external crate dependencies
    pub fn hash(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            let idx = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[idx];
        }
        crc ^ 0xFFFF_FFFF
    }

    // CRC32C（Castagnoli）查找表
    const CRC32_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let polynomial: u32 = 0x82F6_3B78; // CRC32C
        let mut i = 0usize;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ polynomial;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn wal_write_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write records
        {
            let mut writer = WalWriter::open(&wal_path, true).unwrap();
            writer.append(&WalRecord::Insert {
                collection: "test".to_string(),
                document: Document::new("doc_1", vec![1.0, 2.0, 3.0]),
            }).unwrap();
            writer.append(&WalRecord::Delete {
                collection: "test".to_string(),
                doc_id: DocumentId::from("doc_2"),
            }).unwrap();
        }

        // Replay records
        let reader = WalReader::new(&wal_path);
        let records = reader.replay().unwrap();

        assert_eq!(records.len(), 2);
        assert!(matches!(records[0], WalRecord::Insert { .. }));
        assert!(matches!(records[1], WalRecord::Delete { .. }));
    }

    #[test]
    fn wal_handles_empty_file() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("empty.wal");
        // File does not exist
        let reader = WalReader::new(&wal_path);
        let records = reader.replay().unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn crc32_consistency() {
        let data = b"hello openvec";
        let h1 = crc32fast::hash(data);
        let h2 = crc32fast::hash(data);
        assert_eq!(h1, h2);
        // CRC differs after modifying data
        let h3 = crc32fast::hash(b"hello openvec!");
        assert_ne!(h1, h3);
    }
}
