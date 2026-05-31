use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::handlers::HttpResult;
use crate::state::AppState;
use openvec_core::types::DistanceMetric;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Collection metadata response
#[derive(Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub vector_fields: Vec<VectorFieldInfo>,
    pub index_types: HashMap<String, &'static str>,
    pub doc_count: usize,
}

#[derive(Serialize)]
pub struct VectorFieldInfo {
    pub name: String,
    pub dimension: usize,
    pub distance: String,
}

/// Collection creation request
#[derive(Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(default)]
    pub metric: DistanceMetric,
    pub fulltext_fields: Option<Vec<String>>,
}

/// Drop collection response
#[derive(Serialize)]
pub struct DropCollectionResponse {
    pub dropped: bool,
}

/// GET /health
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// GET /collections
pub async fn list_collections(
    State(state): State<AppState>,
) -> HttpResult<Vec<CollectionInfo>> {
    let db = state.db.read();
    let names = db.list_collections();
    let mut infos = Vec::with_capacity(names.len());

    for name in names {
        if let Ok(coll) = db.get_collection(&name) {
            let fields = coll.config().schema.vector_fields.iter().map(|vf| {
                VectorFieldInfo {
                    name: vf.name.clone(),
                    dimension: vf.dimension,
                    distance: vf.distance.to_string(),
                }
            }).collect();

            infos.push(CollectionInfo {
                name: coll.name().to_string(),
                vector_fields: fields,
                index_types: coll.index_types(),
                doc_count: coll.doc_count(),
            });
        }
    }

    Ok(Json(infos))
}

/// POST /collections
pub async fn create_collection(
    State(state): State<AppState>,
    Json(payload): Json<CreateCollectionRequest>,
) -> HttpResult<CollectionInfo> {
    let db = state.db.read();
    
    let coll = if let Some(ref ft_fields) = payload.fulltext_fields {
        let mut schema = openvec_core::types::Schema::new().add_vector_field(
            openvec_core::types::VectorField::new("default", payload.dimension).with_distance(payload.metric)
        );
        for field in ft_fields {
            schema = schema.add_scalar_field(openvec_core::types::ScalarField::full_text(field));
        }
        db.create_collection_with_schema(payload.name, schema)?
    } else {
        db.create_collection(payload.name, payload.dimension, payload.metric)?
    };

    let fields = coll.config().schema.vector_fields.iter().map(|vf| {
        VectorFieldInfo {
            name: vf.name.clone(),
            dimension: vf.dimension,
            distance: vf.distance.to_string(),
        }
    }).collect();

    let info = CollectionInfo {
        name: coll.name().to_string(),
        vector_fields: fields,
        index_types: coll.index_types(),
        doc_count: coll.doc_count(),
    };

    Ok(Json(info))
}

/// DELETE /collections/:name
pub async fn drop_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> HttpResult<DropCollectionResponse> {
    let db = state.db.read();
    let dropped = db.drop_collection(&name)?;
    Ok(Json(DropCollectionResponse { dropped }))
}
