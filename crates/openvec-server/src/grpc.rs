use std::sync::Arc;
use tonic::{Request as TonicRequest, Response as TonicResponse, Status};
use parking_lot::RwLock;

use openvec_core::OpenVec;
use openvec_core::types::{
    Document as CoreDocument, DocumentId as CoreDocumentId, DistanceMetric as CoreMetric,
    SearchRequest as CoreSearchRequest, ScalarValue as CoreScalarValue, Schema as CoreSchema,
    VectorField as CoreVectorField, ScalarField as CoreScalarField
};

pub mod pb {
    tonic::include_proto!("openvec");
}

use pb::open_vec_service_server::OpenVecService;
use pb::{
    CreateCollectionRequest, CollectionMetaResponse, DropCollectionRequest, DropCollectionResponse,
    ListCollectionsRequest, ListCollectionsResponse, InsertDocumentRequest, InsertDocumentResponse,
    BatchInsertRequest, BatchInsertResponse, GetDocumentRequest, GetDocumentResponse,
    DeleteDocumentRequest, DeleteDocumentResponse, SearchRequest, SearchResponse,
    Document as ProtoDocument, SearchResult as ProtoSearchResult, ScalarValue as ProtoScalarValue
};

pub struct OpenVecGrpcService {
    db: Arc<RwLock<OpenVec>>,
}

impl OpenVecGrpcService {
    pub fn new(db: Arc<RwLock<OpenVec>>) -> Self {
        Self { db }
    }
}

// ─────────────────────────────────────────────
//  Type Conversions
// ─────────────────────────────────────────────

fn to_core_metric(m: &str) -> CoreMetric {
    match m.to_lowercase().as_str() {
        "l2" => CoreMetric::L2,
        "cosine" => CoreMetric::Cosine,
        "dot" | "dot_product" => CoreMetric::DotProduct,
        _ => CoreMetric::Cosine,
    }
}

fn to_proto_scalar(val: &CoreScalarValue) -> ProtoScalarValue {
    let mut pb_val = ProtoScalarValue::default();
    match val {
        CoreScalarValue::Int(i) => pb_val.value = Some(pb::scalar_value::Value::IntValue(*i)),
        CoreScalarValue::Float(f) => pb_val.value = Some(pb::scalar_value::Value::FloatValue(*f)),
        CoreScalarValue::Text(t) => pb_val.value = Some(pb::scalar_value::Value::TextValue(t.clone())),
        CoreScalarValue::Bool(b) => pb_val.value = Some(pb::scalar_value::Value::BoolValue(*b)),
        CoreScalarValue::Null => pb_val.value = Some(pb::scalar_value::Value::NullValue(true)),
    }
    pb_val
}

fn to_core_scalar(pb_val: &ProtoScalarValue) -> CoreScalarValue {
    match &pb_val.value {
        Some(pb::scalar_value::Value::IntValue(i)) => CoreScalarValue::Int(*i),
        Some(pb::scalar_value::Value::FloatValue(f)) => CoreScalarValue::Float(*f),
        Some(pb::scalar_value::Value::TextValue(t)) => CoreScalarValue::Text(t.clone()),
        Some(pb::scalar_value::Value::BoolValue(b)) => CoreScalarValue::Bool(*b),
        _ => CoreScalarValue::Null,
    }
}

fn to_proto_doc(doc: CoreDocument) -> ProtoDocument {
    let vector = doc.default_vector().cloned().unwrap_or_default();
    let payload = doc.payload.into_iter()
        .map(|(k, v)| (k, to_proto_scalar(&v)))
        .collect();
    
    ProtoDocument {
        id: doc.id.to_string(),
        vector,
        payload,
    }
}

fn to_core_doc(pb_doc: ProtoDocument) -> CoreDocument {
    let mut doc = CoreDocument::new(CoreDocumentId::from(pb_doc.id), pb_doc.vector);
    doc.payload = pb_doc.payload.into_iter()
        .map(|(k, v)| (k, to_core_scalar(&v)))
        .collect();
    doc
}

// ─────────────────────────────────────────────
//  Service Implementation
// ─────────────────────────────────────────────

#[tonic::async_trait]
impl OpenVecService for OpenVecGrpcService {
    async fn create_collection(
        &self,
        request: TonicRequest<CreateCollectionRequest>,
    ) -> Result<TonicResponse<CollectionMetaResponse>, Status> {
        let req = request.into_inner();
        let db = self.db.read();
        
        let metric = to_core_metric(&req.metric);
        let mut schema = CoreSchema::new()
            .add_vector_field(CoreVectorField::new("default", req.dimension as usize).with_distance(metric));
            
        for field in &req.fulltext_fields {
            schema = schema.add_scalar_field(CoreScalarField::full_text(field));
        }

        db.create_collection_with_schema(&req.name, schema)
            .map_err(|e| Status::already_exists(e.to_string()))?;

        Ok(TonicResponse::new(CollectionMetaResponse {
            name: req.name,
            dimension: req.dimension,
            metric: req.metric,
            fulltext_fields: req.fulltext_fields,
        }))
    }

    async fn drop_collection(
        &self,
        request: TonicRequest<DropCollectionRequest>,
    ) -> Result<TonicResponse<DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let db = self.db.read();
        
        let success = db.drop_collection(&req.name)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        Ok(TonicResponse::new(DropCollectionResponse { success }))
    }

    async fn list_collections(
        &self,
        _request: TonicRequest<ListCollectionsRequest>,
    ) -> Result<TonicResponse<ListCollectionsResponse>, Status> {
        let db = self.db.read();
        let collections_names = db.list_collections();
        
        let mut collections = Vec::new();
        for name in collections_names {
            if let Ok(coll) = db.get_collection(&name) {
                let schema = coll.config().schema.default_vector_field();
                let dim = schema.map(|s| s.dimension as u32).unwrap_or(0);
                let metric = schema.map(|s| s.distance.as_str().to_string()).unwrap_or_default();
                let fulltext_fields = coll.config().schema.scalar_fields.iter()
                    .map(|f| f.name.clone())
                    .collect();
                    
                collections.push(CollectionMetaResponse {
                    name,
                    dimension: dim,
                    metric,
                    fulltext_fields,
                });
            }
        }
        
        Ok(TonicResponse::new(ListCollectionsResponse { collections }))
    }

    async fn insert_document(
        &self,
        request: TonicRequest<InsertDocumentRequest>,
    ) -> Result<TonicResponse<InsertDocumentResponse>, Status> {
        let req = request.into_inner();
        let doc = req.document.ok_or_else(|| Status::invalid_argument("Missing document"))?;
        let core_doc = to_core_doc(doc);
        let id_str = core_doc.id.to_string();
        
        let db = self.db.read();
        let coll = db.get_collection(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;
        drop(db);
            
        coll.insert(core_doc)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        Ok(TonicResponse::new(InsertDocumentResponse { id: id_str }))
    }

    async fn batch_insert(
        &self,
        request: TonicRequest<BatchInsertRequest>,
    ) -> Result<TonicResponse<BatchInsertResponse>, Status> {
        let req = request.into_inner();
        let core_docs: Vec<CoreDocument> = req.documents.into_iter()
            .map(to_core_doc)
            .collect();
        let ids_str: Vec<String> = core_docs.iter().map(|d| d.id.to_string()).collect();
        
        let db = self.db.read();
        let coll = db.get_collection(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;
        drop(db);
            
        coll.batch_insert(core_docs)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        Ok(TonicResponse::new(BatchInsertResponse { ids: ids_str }))
    }

    async fn get_document(
        &self,
        request: TonicRequest<GetDocumentRequest>,
    ) -> Result<TonicResponse<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        let doc_id = CoreDocumentId::from(req.id);
        
        let db = self.db.read();
        let coll = db.get_collection_read(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;
        drop(db);
            
        let doc_opt = coll.get(&doc_id)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        match doc_opt {
            Some(doc) => Ok(TonicResponse::new(GetDocumentResponse {
                found: true,
                document: Some(to_proto_doc(doc)),
            })),
            None => Ok(TonicResponse::new(GetDocumentResponse {
                found: false,
                document: None,
            })),
        }
    }

    async fn delete_document(
        &self,
        request: TonicRequest<DeleteDocumentRequest>,
    ) -> Result<TonicResponse<DeleteDocumentResponse>, Status> {
        let req = request.into_inner();
        let doc_id = CoreDocumentId::from(req.id);
        
        let db = self.db.read();
        let coll = db.get_collection(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;
        drop(db);
            
        let success = coll.delete(&doc_id)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        Ok(TonicResponse::new(DeleteDocumentResponse { success }))
    }

    async fn search(
        &self,
        request: TonicRequest<SearchRequest>,
    ) -> Result<TonicResponse<SearchResponse>, Status> {
        let req = request.into_inner();
        let limit = req.limit as usize;
        
        let mut core_req = CoreSearchRequest::new(req.vector, limit);
        if !req.vector_field.is_empty() {
            core_req = core_req.with_vector_field(req.vector_field);
        }
        if let Some(ef_val) = req.ef {
            core_req = core_req.with_ef(ef_val as usize);
        }
        if let Some(h_q) = req.hybrid_query {
            core_req = core_req.with_hybrid_query(h_q);
        }
        if req.vector_weight.is_some() || req.text_weight.is_some() {
            core_req = core_req.with_weights(
                req.vector_weight.unwrap_or(1.0),
                req.text_weight.unwrap_or(1.0),
            );
        }

        let db = self.db.read();
        let coll = db.get_collection_read(&req.collection)
            .map_err(|e| Status::not_found(e.to_string()))?;
        drop(db);
            
        let results = coll.search(&core_req)
            .map_err(|e| Status::internal(e.to_string()))?;
            
        let proto_results = results.into_iter().map(|res| {
            let payload = res.payload.unwrap_or_default().into_iter()
                .map(|(k, v)| (k, to_proto_scalar(&v)))
                .collect();
            ProtoSearchResult {
                id: res.id.to_string(),
                score: res.score,
                payload,
            }
        }).collect();

        Ok(TonicResponse::new(SearchResponse { results: proto_results }))
    }
}
