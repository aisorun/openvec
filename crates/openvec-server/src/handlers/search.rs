use axum::{
    extract::{Path, State},
    Json,
};
use serde::Deserialize;

use crate::handlers::HttpResult;
use crate::state::AppState;
use openvec_core::types::{Filter, SearchRequest, SearchResult};

/// JSON search query payload
#[derive(Deserialize)]
pub struct SearchQueryRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_vector_field")]
    pub vector_field: String,
    pub limit: usize,
    pub filter: Option<Filter>,
    pub ef: Option<usize>,
    #[serde(default = "default_true")]
    pub with_payload: bool,
    pub hybrid_query: Option<String>,
    pub vector_weight: Option<f32>,
    pub text_weight: Option<f32>,
}

fn default_vector_field() -> String {
    "default".to_string()
}

fn default_true() -> bool {
    true
}

impl SearchQueryRequest {
    pub fn to_search_request(self) -> SearchRequest {
        let mut req = SearchRequest::new(self.vector, self.limit)
            .with_vector_field(self.vector_field);

        if let Some(f) = self.filter {
            req = req.with_filter(f);
        }

        if let Some(ef) = self.ef {
            req = req.with_ef(ef);
        }

        if let Some(ref h_q) = self.hybrid_query {
            req = req.with_hybrid_query(h_q.clone());
        }

        if self.vector_weight.is_some() || self.text_weight.is_some() {
            req = req.with_weights(
                self.vector_weight.unwrap_or(1.0),
                self.text_weight.unwrap_or(1.0),
            );
        }

        if !self.with_payload {
            req = req.without_payload();
        }

        req
    }
}

/// POST /collections/:name/search
pub async fn search(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<SearchQueryRequest>,
) -> HttpResult<Vec<SearchResult>> {
    let db = state.db.read();
    let coll = db.get_collection_read(&collection_name)?;
    drop(db);

    let search_req = payload.to_search_request();
    let results = coll.search(&search_req)?;

    Ok(Json(results))
}
