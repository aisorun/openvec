/// Storage Engine Module
///
/// Implements an LSM-Tree variant architecture:
/// - WAL (Write-Ahead Log): Guarantees durability
/// - MemTable: Memory write buffer
/// - Segment: Immutable persistent files

pub mod wal;
pub mod segment;
pub mod engine;

pub use engine::StorageEngine;
