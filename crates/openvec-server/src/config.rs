use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub hnsw_defaults: HnswDefaults,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StorageConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_false")]
    pub wal_sync: bool,
    #[serde(default = "default_true")]
    pub use_compression: bool,
    #[serde(default = "default_segment_size")]
    pub segment_max_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HnswDefaults {
    #[serde(default = "default_m")]
    pub m: usize,
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
            hnsw_defaults: HnswDefaults::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            grpc_port: default_grpc_port(),
            log_level: default_log_level(),
            api_key: None,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            wal_sync: default_false(),
            use_compression: default_true(),
            segment_max_size: default_segment_size(),
        }
    }
}

impl Default for HnswDefaults {
    fn default() -> Self {
        Self {
            m: default_m(),
            ef_construction: default_ef_construction(),
            ef_search: default_ef_search(),
        }
    }
}

// Default field generators for serde
fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8000
}

fn default_grpc_port() -> u16 {
    9000
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_data_dir() -> String {
    "./openvec_data".to_string()
}

fn default_false() -> bool {
    false
}

fn default_true() -> bool {
    true
}

fn default_segment_size() -> usize {
    10 * 1024 * 1024 // 10MB
}

fn default_m() -> usize {
    16
}

fn default_ef_construction() -> usize {
    100
}

fn default_ef_search() -> usize {
    50
}

impl Config {
    /// Loads configuration from a given file path.
    /// If the file does not exist, it quietly returns the default settings.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let p = path.as_ref();
        if !p.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(p)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config_loading() {
        // Loading non-existing file should yield default values
        let config = Config::load_from_file("non_existent_file_xyz.toml").unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.server.grpc_port, 9000);
        assert_eq!(config.storage.data_dir, "./openvec_data");
        assert_eq!(config.storage.wal_sync, false);
        assert_eq!(config.hnsw_defaults.m, 16);
    }

    #[test]
    fn test_toml_parsing() {
        let toml_str = r#"
            [server]
            host = "0.0.0.0"
            port = 8888
            api_key = "secret_key_123"

            [storage]
            data_dir = "/tmp/openvec"
            wal_sync = true

            [hnsw_defaults]
            m = 32
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", toml_str).unwrap();

        let config = Config::load_from_file(temp_file.path()).unwrap();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8888);
        assert_eq!(config.server.grpc_port, 9000); // Default preserved
        assert_eq!(config.server.api_key, Some("secret_key_123".to_string()));
        assert_eq!(config.storage.data_dir, "/tmp/openvec");
        assert_eq!(config.storage.wal_sync, true);
        assert_eq!(config.hnsw_defaults.m, 32);
        assert_eq!(config.hnsw_defaults.ef_construction, 100); // Default preserved
    }
}
