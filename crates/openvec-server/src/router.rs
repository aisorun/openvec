use axum::{
    routing::{delete, get, post},
    Router,
    extract::State,
    middleware::{self, Next},
    http::{Request, StatusCode},
    response::Response,
    body::Body,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;

/// Authentication middleware for REST API
async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(ref expected_key) = state.api_key {
        let api_key = req.headers()
            .get("X-API-Key")
            .and_then(|value| value.to_str().ok())
            .or_else(|| {
                req.headers()
                    .get("Authorization")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|auth_str| auth_str.strip_prefix("Bearer "))
            });

        if let Some(key) = api_key {
            if key == expected_key {
                return Ok(next.run(req).await);
            }
        }
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    Ok(next.run(req).await)
}

/// Constructs the entire Axum router with middlewares
pub fn build_router(state: AppState) -> Router {
    // Setup CORS
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let api_routes = Router::new()
        // Collections management
        .route("/collections", get(handlers::collection::list_collections).post(handlers::collection::create_collection))
        .route("/collections/{name}", delete(handlers::collection::drop_collection))
        // Document management
        .route("/collections/{name}/insert", post(handlers::document::insert_document))
        .route("/collections/{name}/batch_insert", post(handlers::document::batch_insert))
        .route("/collections/{name}/documents/{id}", get(handlers::document::get_document).delete(handlers::document::delete_document))
        // Vector Query
        .route("/collections/{name}/search", post(handlers::search::search))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    Router::new()
        // Health check (Exempt from auth)
        .route("/health", get(handlers::collection::health))
        .merge(api_routes)
        // Middlewares
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt; // for `oneshot`
    use serde_json::{json, Value};
    use tempfile::tempdir;
    use openvec_core::OpenVec;

    #[tokio::test]
    async fn test_server_rest_api_flow() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();
        let state = AppState {
            db: std::sync::Arc::new(parking_lot::RwLock::new(db)),
            api_key: None,
        };
        let app = build_router(state);

        // 1. Health check
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json_body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json_body["status"], "healthy");

        // 2. Create Collection
        let create_payload = json!({
            "name": "articles",
            "dimension": 3,
            "metric": "l2"
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(create_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json_body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json_body["name"], "articles");

        // 3. List Collections
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json_body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json_body.as_array().unwrap().len(), 1);
        assert_eq!(json_body[0]["name"], "articles");

        // 4. Insert Document
        let doc_payload = json!({
            "id": "doc_1",
            "vector": [1.0, 2.0, 3.0],
            "payload": {
                "title": "Introduction to Rust"
            }
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/articles/insert")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(doc_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // 5. Search nearest neighbor
        let search_payload = json!({
            "vector": [1.0, 2.0, 3.1],
            "limit": 1
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/articles/search")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(search_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json_body: Value = serde_json::from_slice(&body).unwrap();
        let results = json_body.as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], "doc_1");
        assert!(results[0]["payload"]["title"].is_string());

        // 6. Delete Collection
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/articles")
                    .method("DELETE")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_server_hybrid_search_flow() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();
        let state = AppState {
            db: std::sync::Arc::new(parking_lot::RwLock::new(db)),
            api_key: None,
        };
        let app = build_router(state);

        // 1. Create Collection with full-text fields
        let create_payload = json!({
            "name": "books",
            "dimension": 2,
            "metric": "l2",
            "fulltext_fields": ["description"]
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(create_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // 2. Insert Document with text description
        let doc_payload = json!({
            "id": "book_rust",
            "vector": [1.0, 1.0],
            "payload": {
                "description": "The definitive guide to Rust programming and databases."
            }
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/books/insert")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(doc_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // 3. Perform Hybrid Search
        let search_payload = json!({
            "vector": [1.0, 1.1],
            "limit": 5,
            "hybrid_query": "Rust programming database"
        });
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections/books/search")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(Body::from(search_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json_body: Value = serde_json::from_slice(&body).unwrap();
        let results = json_body.as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], "book_rust");
    }

    #[tokio::test]
    async fn test_server_api_key_auth_flow() {
        let dir = tempdir().unwrap();
        let db = OpenVec::open(dir.path()).unwrap();
        let state = AppState {
            db: std::sync::Arc::new(parking_lot::RwLock::new(db)),
            api_key: Some("test-secret-key".to_string()),
        };
        let app = build_router(state);

        // 1. Health check (Must succeed without any API Key because it is exempt)
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // 2. Request to protected route without API Key (Must fail with 401 Unauthorized)
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // 3. Request to protected route with wrong API Key (Must fail with 401 Unauthorized)
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("GET")
                    .header("X-API-Key", "wrong-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // 4. Request to protected route with correct API Key in X-API-Key (Must succeed)
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("GET")
                    .header("X-API-Key", "test-secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // 5. Request to protected route with correct API Key in Authorization Bearer (Must succeed)
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .method("GET")
                    .header("Authorization", "Bearer test-secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
