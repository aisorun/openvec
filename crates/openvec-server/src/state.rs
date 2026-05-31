use std::sync::Arc;
use parking_lot::RwLock;
use openvec_core::OpenVec;

/// Shared application state for Axum
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<OpenVec>>,
    pub api_key: Option<String>,
}
