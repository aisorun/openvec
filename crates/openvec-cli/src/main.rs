use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::PathBuf;

use openvec_core::OpenVec;
use openvec_core::types::{DistanceMetric, Document, DocumentId, ScalarField, ScalarValue, Schema, SearchRequest, VectorField};

/// OpenVec Command Line Interface
#[derive(Parser, Debug)]
#[command(name = "openvec", version, about = "OpenVec Vector Database Command Line Interface", long_about = None)]
struct Cli {
    /// Path to the database data directory
    #[arg(short, long, global = true, default_value = "./openvec_data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List all registered collections
    List,

    /// Create a new collection
    Create {
        /// Collection name
        name: String,

        /// Vector dimension size
        #[arg(short = 'D', long)]
        dim: usize,

        /// Distance metric (l2, cosine, dot)
        #[arg(short, long, value_enum, default_value_t = CliMetric::Cosine)]
        metric: CliMetric,

        /// Optional full-text BM25 fields (can be specified multiple times, e.g. --fulltext-field content --fulltext-field title)
        #[arg(long = "fulltext-field", value_name = "FIELD")]
        fulltext_fields: Vec<String>,
    },

    /// Drop an existing collection
    Drop {
        /// Collection name
        name: String,
    },

    /// Insert a document into a collection
    Insert {
        /// Collection name
        collection: String,

        /// Unique document ID (auto-generated if not specified)
        #[arg(long)]
        id: Option<String>,

        /// Vector embedding formatted as a JSON array (e.g. '[0.1, 0.2, 0.3]')
        #[arg(short, long)]
        vector: String,

        /// Optional scalar payload formatted as a JSON object (e.g. '{"category": "tech"}')
        #[arg(short, long)]
        payload: Option<String>,
    },

    /// Search for nearest neighbors in a collection
    Search {
        /// Collection name
        collection: String,

        /// Query vector formatted as a JSON array (e.g. '[0.1, 0.2, 0.3]')
        #[arg(short, long)]
        vector: String,

        /// Max number of results to return
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        /// Optional HNSW ef_search parameter
        #[arg(long)]
        ef: Option<usize>,

        /// Optional lexical text search query for RRF hybrid search
        #[arg(short = 'H', long)]
        hybrid: Option<String>,

        /// Optional vector weight for hybrid search
        #[arg(long, default_value_t = 1.0)]
        vector_weight: f32,

        /// Optional text weight for hybrid search
        #[arg(long, default_value_t = 1.0)]
        text_weight: f32,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum CliMetric {
    L2,
    Cosine,
    Dot,
}

impl From<CliMetric> for DistanceMetric {
    fn from(m: CliMetric) -> Self {
        match m {
            CliMetric::L2 => DistanceMetric::L2,
            CliMetric::Cosine => DistanceMetric::Cosine,
            CliMetric::Dot => DistanceMetric::DotProduct,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize database instance locally
    let db = OpenVec::open(&cli.data_dir)?;

    match cli.command {
        Commands::List => {
            let collections = db.list_collections();
            if collections.is_empty() {
                println!("No collections registered.");
                return Ok(());
            }

            println!("{:<20} | {:<10} | {:<15} | {:<10} | {:<10}", "Collection", "Dimension", "Metric", "Index", "Docs");
            println!("{}", "-".repeat(75));

            for name in collections {
                if let Ok(coll) = db.get_collection(&name) {
                    let schema = coll.config().schema.default_vector_field();
                    let dim_str = schema.map_or("N/A".to_string(), |s| s.dimension.to_string());
                    let metric_str = schema.map_or("N/A".to_string(), |s| s.distance.to_string());
                    let index_types = coll.index_types();
                    let index_str = index_types.get("default").copied().unwrap_or("flat");
                    let doc_count = coll.doc_count();

                    println!("{:<20} | {:<10} | {:<15} | {:<10} | {:<10}", name, dim_str, metric_str, index_str, doc_count);
                }
            }
        }
        Commands::Create { name, dim, metric, fulltext_fields } => {
            if fulltext_fields.is_empty() {
                db.create_collection(&name, dim, metric.into())?;
                println!("Successfully created collection '{}' with dimension {} ({} metric).", name, dim, DistanceMetric::from(metric));
            } else {
                let mut schema = Schema::new().add_vector_field(
                    VectorField::new("default", dim).with_distance(metric.into())
                );
                for field in &fulltext_fields {
                    schema = schema.add_scalar_field(ScalarField::full_text(field));
                }
                db.create_collection_with_schema(&name, schema)?;
                println!(
                    "Successfully created collection '{}' with dimension {} ({} metric) and fulltext fields: [{}].",
                    name, dim, DistanceMetric::from(metric),
                    fulltext_fields.join(", ")
                );
            }
        }
        Commands::Drop { name } => {
            let dropped = db.drop_collection(&name)?;
            if dropped {
                println!("Successfully dropped collection '{}'.", name);
            } else {
                println!("Collection '{}' did not exist.", name);
            }
        }
        Commands::Insert { collection, id, vector, payload } => {
            let parsed_vector: Vec<f32> = serde_json::from_str(&vector)
                .map_err(|e| anyhow::anyhow!("Invalid vector JSON array format: {}", e))?;

            let parsed_payload: HashMap<String, ScalarValue> = if let Some(p_str) = payload {
                serde_json::from_str(&p_str)
                    .map_err(|e| anyhow::anyhow!("Invalid payload JSON object format: {}", e))?
            } else {
                HashMap::new()
            };

            let coll = db.get_collection(&collection)?;
            let doc_id = id.map(DocumentId::from).unwrap_or_else(DocumentId::new_random);
            let id_str = doc_id.to_string();

            let mut doc = Document::new(doc_id, parsed_vector);
            doc.payload = parsed_payload;

            coll.insert(doc)?;
            println!("Successfully inserted document ID '{}' into collection '{}'.", id_str, collection);
        }
        Commands::Search { collection, vector, limit, ef, hybrid, vector_weight, text_weight } => {
            let parsed_vector: Vec<f32> = serde_json::from_str(&vector)
                .map_err(|e| anyhow::anyhow!("Invalid vector JSON array format: {}", e))?;

            let coll = db.get_collection(&collection)?;

            let mut req = SearchRequest::new(parsed_vector, limit);
            if let Some(ef_val) = ef {
                req = req.with_ef(ef_val);
            }
            if let Some(h_q) = hybrid {
                req = req.with_hybrid_query(h_q);
                req = req.with_weights(vector_weight, text_weight);
            }

            let results = coll.search(&req)?;

            if results.is_empty() {
                println!("No matching results found.");
                return Ok(());
            }

            println!("{:<6} | {:<36} | {:<10} | {:<30}", "Rank", "Document ID", "Score", "Payload");
            println!("{}", "-".repeat(90));

            for (idx, res) in results.into_iter().enumerate() {
                let payload_str = match res.payload {
                    Some(ref p) if !p.is_empty() => serde_json::to_string(p).unwrap_or_default(),
                    _ => "{}".to_string(),
                };
                // Truncate payload string for pretty formatting
                let truncated_payload = if payload_str.len() > 30 {
                    format!("{}...", &payload_str[..27])
                } else {
                    payload_str
                };

                println!("{:<6} | {:<36} | {:<10.6} | {:<30}", idx + 1, res.id.to_string(), res.score, truncated_payload);
            }
        }
    }

    Ok(())
}
