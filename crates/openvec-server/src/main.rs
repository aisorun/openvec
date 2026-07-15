use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, warn};

use openvec_core::OpenVec;

mod config;
mod handlers;
mod router;
mod state;
mod grpc;

use state::AppState;
use grpc::{OpenVecGrpcService, pb};

/// OpenVec Server CLI configuration
#[derive(Parser, Debug)]
#[command(name = "openvec-server", version, about = "OpenVec Vector Database HTTP/gRPC Server")]
struct Args {
    /// Path to configuration file
    #[arg(short = 'c', long)]
    config: Option<String>,

    /// Host address to bind to (overrides config)
    #[arg(short = 'H', long)]
    host: Option<String>,

    /// Port to listen on for HTTP REST (overrides config)
    #[arg(short, long)]
    port: Option<u16>,

    /// Port to listen on for gRPC (overrides config)
    #[arg(short = 'g', long)]
    grpc_port: Option<u16>,

    /// Data directory path for database storage (overrides config)
    #[arg(short, long)]
    data_dir: Option<String>,

    /// API Key for HTTP/gRPC authentication (overrides config)
    #[arg(short = 'k', long)]
    api_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Load configuration from file (defaults to openvec.toml)
    let config_path = args.config.clone().unwrap_or_else(|| "openvec.toml".to_string());
    
    let mut config = match config::Config::load_from_file(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            warn!("Failed to parse config file: {}. Using default values. Error: {}", config_path, e);
            config::Config::default()
        }
    };

    // 2. Override configuration with command-line arguments if specified
    if let Some(host) = args.host {
        config.server.host = host;
    }
    if let Some(port) = args.port {
        config.server.port = port;
    }
    if let Some(grpc_port) = args.grpc_port {
        config.server.grpc_port = grpc_port;
    }
    if let Some(data_dir) = args.data_dir {
        config.storage.data_dir = data_dir;
    }
    if let Some(api_key) = args.api_key {
        config.server.api_key = Some(api_key);
    }

    // 3. Initialize logging (respecting config log_level)
    let log_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| {
        format!(
            "openvec_server={},tower_http={}",
            config.server.log_level, config.server.log_level
        )
    });
    
    tracing_subscriber::fmt()
        .with_env_filter(log_filter)
        .init();

    info!("Starting OpenVec Server v{}", env!("CARGO_PKG_VERSION"));
    info!("Storage directory: {}", config.storage.data_dir);

    // Open/Initialize the database
    let db = OpenVec::open(&config.storage.data_dir)?
        .with_wal_sync(config.storage.wal_sync)
        .with_compress(config.storage.use_compression);
    for name in db.list_collections() {
        if let Err(e) = db.get_collection(&name) {
            warn!("Failed to preload collection '{}': {}", name, e);
        } else {
            info!("Successfully preloaded collection '{}' into memory", name);
        }
    }
    let shared_db = Arc::new(RwLock::new(db));

    let state = AppState {
        db: shared_db.clone(),
        api_key: config.server.api_key.clone(),
    };

    // 1. Build and spawn the Tonic gRPC Server
    let grpc_addr_str = format!("{}:{}", config.server.host, config.server.grpc_port);
    let grpc_addr: SocketAddr = grpc_addr_str.parse()?;
    let grpc_db = shared_db.clone();
    let api_key = config.server.api_key.clone();
    
    tokio::spawn(async move {
        let grpc_service = OpenVecGrpcService::new(grpc_db);
        info!("OpenVec gRPC Server listening on http://{}", grpc_addr);
        
        let mut builder = tonic::transport::Server::builder();
        
        if let Some(expected_key) = api_key {
            let auth_interceptor = move |req: tonic::Request<()>| {
                let metadata = req.metadata();
                let client_key = metadata.get("x-api-key")
                    .or_else(|| metadata.get("authorization"))
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| {
                        if s.starts_with("Bearer ") {
                            Some(s.trim_start_matches("Bearer "))
                        } else {
                            Some(s)
                        }
                    });

                if let Some(key) = client_key {
                    if key == expected_key {
                        return Ok(req);
                    }
                }
                Err(tonic::Status::unauthenticated("Invalid or missing API key"))
            };
            
            builder
                .add_service(pb::open_vec_service_server::OpenVecServiceServer::with_interceptor(grpc_service, auth_interceptor))
                .serve(grpc_addr)
                .await
                .unwrap();
        } else {
            builder
                .add_service(pb::open_vec_service_server::OpenVecServiceServer::new(grpc_service))
                .serve(grpc_addr)
                .await
                .unwrap();
        }
    });

    // 2. Build and run the Axum router with REST endpoints
    let app = router::build_router(state);

    // Bind and listen for HTTP
    let addr_str = format!("{}:{}", config.server.host, config.server.port);
    let addr: SocketAddr = addr_str.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("OpenVec HTTP REST Server listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

