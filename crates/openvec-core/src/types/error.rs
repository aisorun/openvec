use thiserror::Error;

/// OpenVec unified error type
#[derive(Debug, Error)]
pub enum Error {
    // ── Collection errors ──────────────────────────────
    #[error("Collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Invalid collection name '{0}': {1}")]
    InvalidCollectionName(String, String),

    // ── Document errors ────────────────────────────────
    #[error("Document '{0}' not found")]
    DocumentNotFound(String),

    #[error("Duplicate document id '{0}'")]
    DuplicateDocumentId(String),

    // ── Schema/dimension errors ────────────────────────
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Vector field '{0}' not found in schema")]
    VectorFieldNotFound(String),

    #[error("Invalid schema: {0}")]
    InvalidSchema(String),

    // ── Storage errors ─────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Corruption detected: {0}")]
    Corruption(String),

    #[error("WAL replay failed: {0}")]
    WalReplayFailed(String),

    // ── Index errors ───────────────────────────────────
    #[error("Index build failed: {0}")]
    IndexBuildFailed(String),

    #[error("Index is empty")]
    IndexEmpty,

    // ── Serialization errors ───────────────────────────
    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Deserialization failed: {0}")]
    Deserialization(String),

    // ── Query errors ───────────────────────────────────
    #[error("Invalid filter: {0}")]
    InvalidFilter(String),

    #[error("Invalid search request: {0}")]
    InvalidSearchRequest(String),

    // ── Concurrency errors ─────────────────────────────
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    // ── Generic ────────────────────────────────────────
    #[error("Internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// OpenVec Result type alias
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    pub fn corruption(msg: impl Into<String>) -> Self {
        Self::Corruption(msg.into())
    }
}

// serde_json error conversion
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}
