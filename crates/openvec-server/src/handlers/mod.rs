pub mod collection;
pub mod document;
pub mod search;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use openvec_core::types::error::Error;

/// Error response body
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// A wrapper around OpenVec core errors to map them to appropriate HTTP Status Codes
pub struct HttpError(pub Error);

impl From<Error> for HttpError {
    fn from(err: Error) -> Self {
        Self(err)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            Error::CollectionNotFound(name) => (
                StatusCode::NOT_FOUND,
                format!("Collection '{}' not found", name),
            ),
            Error::DocumentNotFound(id) => (
                StatusCode::NOT_FOUND,
                format!("Document '{}' not found", id),
            ),
            Error::CollectionAlreadyExists(name) => (
                StatusCode::CONFLICT,
                format!("Collection '{}' already exists", name),
            ),
            Error::InvalidCollectionName(name, reason) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid collection name '{}': {}", name, reason),
            ),
            Error::DimensionMismatch { expected, got } => (
                StatusCode::BAD_REQUEST,
                format!("Vector dimension mismatch: expected {}, got {}", expected, got),
            ),
            Error::VectorFieldNotFound(field) => (
                StatusCode::BAD_REQUEST,
                format!("Vector field '{}' not found in schema", field),
            ),
            Error::InvalidSchema(reason) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid schema: {}", reason),
            ),
            Error::InvalidFilter(reason) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid filter: {}", reason),
            ),
            Error::InvalidSearchRequest(reason) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid search request: {}", reason),
            ),
            Error::Serialization(reason) | Error::Deserialization(reason) => (
                StatusCode::BAD_REQUEST,
                format!("Data error: {}", reason),
            ),
            other => (
                StatusCode::INTERNAL_SERVER_ERROR,
                other.to_string(),
            ),
        };

        let body = Json(ErrorResponse { error: message });
        (status, body).into_response()
    }
}

/// Simple type alias to make handlers read cleaner
pub type HttpResult<T> = Result<Json<T>, HttpError>;
