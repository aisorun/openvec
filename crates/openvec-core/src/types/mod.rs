/// Public type definitions module

pub mod error;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─────────────────────────────────────────────
//  Identity types
// ─────────────────────────────────────────────

/// Unique identifier for a Collection
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionId(pub String);

impl CollectionId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for CollectionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for CollectionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Unique identifier for a Document
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    /// Generates a new random UUID Document ID
    pub fn new_random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DocumentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for DocumentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// ─────────────────────────────────────────────
//  Vector types
// ─────────────────────────────────────────────

/// Heap-allocated vector (f32)
pub type Vector = Vec<f32>;

/// Vector reference (borrowed slice)
pub type VectorRef<'a> = &'a [f32];

// ─────────────────────────────────────────────
//  Distance metric
// ─────────────────────────────────────────────

/// Metric for calculating vector distance/similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Euclidean distance (L2): smaller values mean more similar
    #[default]
    L2,
    /// Cosine similarity: larger values mean more similar (stored as 1 - cosine_similarity so that smaller is better)
    Cosine,
    /// Dot Product (inner product): larger values mean more similar (stored as negative value so that smaller is better)
    DotProduct,
}

impl DistanceMetric {
    /// Returns the string representation of the metric
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::DotProduct => "dot_product",
        }
    }

    /// Whether a lower score is better (for sorting)
    pub fn lower_is_better(&self) -> bool {
        // After unification, all distances are smaller is better
        true
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─────────────────────────────────────────────
//  Schema types
// ─────────────────────────────────────────────

/// Vector field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
    /// Field name (must be unique within the collection)
    pub name: String,
    /// Dimension of the vector
    pub dimension: usize,
    /// Distance metric
    pub distance: DistanceMetric,
}

impl VectorField {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            distance: DistanceMetric::Cosine,
        }
    }

    pub fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }
}

/// Scalar field types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarFieldType {
    Int,
    Float,
    Text,     // Exact match (keyword)
    FullText, // Full-text search (BM25)
    Bool,
}

/// Scalar field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField {
    pub name: String,
    pub field_type: ScalarFieldType,
    pub indexed: bool,
}

impl ScalarField {
    pub fn int(name: impl Into<String>) -> Self {
        Self { name: name.into(), field_type: ScalarFieldType::Int, indexed: true }
    }

    pub fn float(name: impl Into<String>) -> Self {
        Self { name: name.into(), field_type: ScalarFieldType::Float, indexed: true }
    }

    pub fn text(name: impl Into<String>) -> Self {
        Self { name: name.into(), field_type: ScalarFieldType::Text, indexed: true }
    }

    pub fn full_text(name: impl Into<String>) -> Self {
        Self { name: name.into(), field_type: ScalarFieldType::FullText, indexed: true }
    }

    pub fn bool(name: impl Into<String>) -> Self {
        Self { name: name.into(), field_type: ScalarFieldType::Bool, indexed: true }
    }
}

/// Collection Schema definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Schema {
    pub vector_fields: Vec<VectorField>,
    pub scalar_fields: Vec<ScalarField>,
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_vector_field(mut self, field: VectorField) -> Self {
        self.vector_fields.push(field);
        self
    }

    pub fn add_scalar_field(mut self, field: ScalarField) -> Self {
        self.scalar_fields.push(field);
        self
    }

    /// Find a vector field by name
    pub fn get_vector_field(&self, name: &str) -> Option<&VectorField> {
        self.vector_fields.iter().find(|f| f.name == name)
    }

    /// Get the default vector field (the first one)
    pub fn default_vector_field(&self) -> Option<&VectorField> {
        self.vector_fields.first()
    }
}

// ─────────────────────────────────────────────
//  Document
// ─────────────────────────────────────────────

/// Value of a scalar field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Null,
}

impl ScalarValue {
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self { Some(*v) } else { None }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::Text(v) = self { Some(v.as_str()) } else { None }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self { Some(*v) } else { None }
    }
}

impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self { Self::Int(v) }
}
impl From<i32> for ScalarValue {
    fn from(v: i32) -> Self { Self::Int(v as i64) }
}
impl From<f64> for ScalarValue {
    fn from(v: f64) -> Self { Self::Float(v) }
}
impl From<f32> for ScalarValue {
    fn from(v: f32) -> Self { Self::Float(v as f64) }
}
impl From<String> for ScalarValue {
    fn from(v: String) -> Self { Self::Text(v) }
}
impl From<&str> for ScalarValue {
    fn from(v: &str) -> Self { Self::Text(v.to_string()) }
}
impl From<bool> for ScalarValue {
    fn from(v: bool) -> Self { Self::Bool(v) }
}

/// Document (vectors + metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID (can be user-specified, otherwise auto-generated)
    pub id: DocumentId,
    /// Vector data (supports multiple vector fields)
    pub vectors: HashMap<String, Vector>,
    /// Scalar metadata (JSON compatible)
    pub payload: HashMap<String, ScalarValue>,
}

impl Document {
    /// Creates a new Document with a default vector field name
    pub fn new(id: impl Into<DocumentId>, vector: Vector) -> Self {
        let mut vectors = HashMap::new();
        vectors.insert("default".to_string(), vector);
        Self {
            id: id.into(),
            vectors,
            payload: HashMap::new(),
        }
    }

    /// Creates a new Document with a random ID
    pub fn new_auto(vector: Vector) -> Self {
        Self::new(DocumentId::new_random(), vector)
    }

    pub fn with_payload(mut self, key: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        self.payload.insert(key.into(), value.into());
        self
    }

    pub fn with_named_vector(mut self, name: impl Into<String>, vector: Vector) -> Self {
        self.vectors.insert(name.into(), vector);
        self
    }

    pub fn get_vector(&self, field_name: &str) -> Option<&Vector> {
        self.vectors.get(field_name)
    }

    pub fn default_vector(&self) -> Option<&Vector> {
        self.vectors.get("default")
    }
}

// ─────────────────────────────────────────────
//  Filter types
// ─────────────────────────────────────────────

/// Comparison operator
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
}

/// Single filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub op: CompareOp,
    pub value: ScalarValue,
    pub values: Option<Vec<ScalarValue>>,  // Used for In operations
}

impl FilterCondition {
    pub fn eq(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Eq, value: value.into(), values: None }
    }

    pub fn ne(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Ne, value: value.into(), values: None }
    }

    pub fn gt(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Gt, value: value.into(), values: None }
    }

    pub fn gte(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Gte, value: value.into(), values: None }
    }

    pub fn lt(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Lt, value: value.into(), values: None }
    }

    pub fn lte(field: impl Into<String>, value: impl Into<ScalarValue>) -> Self {
        Self { field: field.into(), op: CompareOp::Lte, value: value.into(), values: None }
    }

    pub fn in_values(field: impl Into<String>, values: Vec<ScalarValue>) -> Self {
        Self {
            field: field.into(),
            op: CompareOp::In,
            value: ScalarValue::Null,
            values: Some(values),
        }
    }

    /// Determines if a document matches this condition
    pub fn matches(&self, doc: &Document) -> bool {
        let field_val = match doc.payload.get(&self.field) {
            Some(v) => v,
            None => return false,
        };

        match &self.op {
            CompareOp::Eq => field_val == &self.value,
            CompareOp::Ne => field_val != &self.value,
            CompareOp::In => {
                if let Some(values) = &self.values {
                    values.contains(field_val)
                } else {
                    false
                }
            }
            CompareOp::Gt | CompareOp::Gte | CompareOp::Lt | CompareOp::Lte => {
                let fv = match field_val.as_float() { Some(v) => v, None => return false };
                let tv = match self.value.as_float() { Some(v) => v, None => return false };
                match self.op {
                    CompareOp::Gt => fv > tv,
                    CompareOp::Gte => fv >= tv,
                    CompareOp::Lt => fv < tv,
                    CompareOp::Lte => fv <= tv,
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Compound filter (AND/OR/NOT)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// Must satisfy all sub-filters
    And(Vec<Filter>),
    /// Must satisfy at least one sub-filter
    Or(Vec<Filter>),
    /// Logical negation
    Not(Box<Filter>),
    /// Single filter condition
    Condition(FilterCondition),
}

impl Filter {
    pub fn and(filters: impl IntoIterator<Item = Filter>) -> Self {
        Self::And(filters.into_iter().collect())
    }

    pub fn or(filters: impl IntoIterator<Item = Filter>) -> Self {
        Self::Or(filters.into_iter().collect())
    }

    pub fn not(filter: Filter) -> Self {
        Self::Not(Box::new(filter))
    }

    pub fn condition(cond: FilterCondition) -> Self {
        Self::Condition(cond)
    }

    /// Determines if a document matches the entire filter
    pub fn matches(&self, doc: &Document) -> bool {
        match self {
            Self::And(filters) => filters.iter().all(|f| f.matches(doc)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(doc)),
            Self::Not(filter) => !filter.matches(doc),
            Self::Condition(cond) => cond.matches(doc),
        }
    }
}

// ─────────────────────────────────────────────
//  Search result types
// ─────────────────────────────────────────────

/// Single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub id: DocumentId,
    /// Distance score (smaller is better, normalized)
    pub score: f32,
    /// Document metadata (optional, depends on query parameters)
    pub payload: Option<HashMap<String, ScalarValue>>,
}

/// Collection of search results
pub type SearchResults = Vec<SearchResult>;

/// Search request
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// Query vector
    pub vector: Vector,
    /// Name of the vector field to search (defaults to "default")
    pub vector_field: String,
    /// Max number of results to return
    pub limit: usize,
    /// Filtering conditions
    pub filter: Option<Filter>,
    /// Whether to return payloads
    pub with_payload: bool,
    /// HNSW ef_search parameter (larger is more accurate but slower)
    pub ef: Option<usize>,
    /// Optional lexical text search query for RRF fusion
    pub hybrid_query: Option<String>,
    /// Optional vector weight for hybrid search (defaults to 1.0)
    pub vector_weight: Option<f32>,
    /// Optional text weight for hybrid search (defaults to 1.0)
    pub text_weight: Option<f32>,
}

impl SearchRequest {
    pub fn new(vector: Vector, limit: usize) -> Self {
        Self {
            vector,
            vector_field: "default".to_string(),
            limit,
            filter: None,
            with_payload: true,
            ef: None,
            hybrid_query: None,
            vector_weight: None,
            text_weight: None,
        }
    }

    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_vector_field(mut self, field: impl Into<String>) -> Self {
        self.vector_field = field.into();
        self
    }

    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    pub fn with_hybrid_query(mut self, query: impl Into<String>) -> Self {
        self.hybrid_query = Some(query.into());
        self
    }

    pub fn with_weights(mut self, vector_weight: f32, text_weight: f32) -> Self {
        self.vector_weight = Some(vector_weight);
        self.text_weight = Some(text_weight);
        self
    }

    pub fn without_payload(mut self) -> Self {
        self.with_payload = false;
        self
    }
}
