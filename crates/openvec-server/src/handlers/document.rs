use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::handlers::{HttpError, HttpResult};
use crate::state::AppState;
use openvec_core::types::{Document, DocumentId, ScalarValue};
use openvec_core::types::error::Error;

/// Individual document insertion payload
#[derive(Deserialize)]
pub struct InsertDocumentRequest {
    pub id: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub vectors: Option<HashMap<String, Vec<f32>>>,
    #[serde(default)]
    pub payload: HashMap<String, ScalarValue>,
}

impl InsertDocumentRequest {
    pub fn to_document(self) -> Result<Document, Error> {
        let id = self.id.map(DocumentId::from).unwrap_or_else(DocumentId::new_random);
        let mut vectors = HashMap::new();

        if let Some(v) = self.vector {
            vectors.insert("default".to_string(), v);
        }

        if let Some(vs) = self.vectors {
            for (k, v) in vs {
                vectors.insert(k, v);
            }
        }

        if vectors.is_empty() {
            return Err(Error::InvalidSearchRequest(
                "Document must contain at least one vector under 'vector' or 'vectors'".to_string()
            ));
        }

        Ok(Document {
            id,
            vectors,
            payload: self.payload,
        })
    }
}

/// Document insertion response
#[derive(Serialize)]
pub struct InsertDocumentResponse {
    pub id: String,
}

/// Batch insertion request
#[derive(Deserialize)]
pub struct BatchInsertRequest {
    pub documents: Vec<InsertDocumentRequest>,
}

/// Batch insertion response
#[derive(Serialize)]
pub struct BatchInsertResponse {
    pub ids: Vec<String>,
}

/// Delete document response
#[derive(Serialize)]
pub struct DeleteDocumentResponse {
    pub deleted: bool,
}

/// POST /collections/:name/insert
pub async fn insert_document(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<InsertDocumentRequest>,
) -> HttpResult<InsertDocumentResponse> {
    let db = state.db.read();
    let coll = db.get_collection(&collection_name)?;
    drop(db);

    let doc = payload.to_document()?;
    let id_str = doc.id.to_string();

    coll.insert(doc)?;
    Ok(Json(InsertDocumentResponse { id: id_str }))
}

/// POST /collections/:name/batch_insert
pub async fn batch_insert(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<BatchInsertRequest>,
) -> HttpResult<BatchInsertResponse> {
    let db = state.db.read();
    let coll = db.get_collection(&collection_name)?;
    drop(db);

    let mut docs = Vec::with_capacity(payload.documents.len());
    let mut ids = Vec::with_capacity(payload.documents.len());

    for req in payload.documents {
        let doc = req.to_document()?;
        ids.push(doc.id.to_string());
        docs.push(doc);
    }

    coll.batch_insert(docs)?;
    Ok(Json(BatchInsertResponse { ids }))
}

/// GET /collections/:name/documents/:id
pub async fn get_document(
    State(state): State<AppState>,
    Path((collection_name, doc_id)): Path<(String, String)>,
) -> HttpResult<Document> {
    let db = state.db.read();
    let coll = db.get_collection(&collection_name)?;
    drop(db);

    let id = DocumentId::from(doc_id.clone());

    match coll.get(&id)? {
        Some(doc) => Ok(Json(doc)),
        None => Err(HttpError(Error::DocumentNotFound(doc_id))),
    }
}

/// DELETE /collections/:name/documents/:id
pub async fn delete_document(
    State(state): State<AppState>,
    Path((collection_name, doc_id)): Path<(String, String)>,
) -> HttpResult<DeleteDocumentResponse> {
    let db = state.db.read();
    let coll = db.get_collection(&collection_name)?;
    drop(db);

    let id = DocumentId::from(doc_id);

    let deleted = coll.delete(&id)?;
    Ok(Json(DeleteDocumentResponse { deleted }))
}
