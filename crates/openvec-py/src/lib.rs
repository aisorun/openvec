use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use openvec_core::{OpenVec, Document, DistanceMetric};
use openvec_core::types::{DocumentId, ScalarValue, SearchRequest};

#[pyclass]
pub struct OpenVecPy {
    inner: OpenVec,
}

#[pymethods]
impl OpenVecPy {
    #[new]
    fn new(data_dir: String) -> PyResult<Self> {
        let db = OpenVec::open(data_dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: db })
    }

    fn create_collection(&self, name: String, dimension: usize, metric: String) -> PyResult<CollectionPy> {
        let distance_metric = match metric.to_lowercase().as_str() {
            "l2" => DistanceMetric::L2,
            "cosine" => DistanceMetric::Cosine,
            "dot" | "dot_product" => DistanceMetric::DotProduct,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid metric. Supported: 'l2', 'cosine', 'dot'")),
        };
        let collection = self.inner.create_collection(&name, dimension, distance_metric)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(CollectionPy { inner: collection })
    }

    fn get_collection(&self, name: String) -> PyResult<CollectionPy> {
        let collection = self.inner.get_collection(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(CollectionPy { inner: collection })
    }

    fn drop_collection(&self, name: String) -> PyResult<bool> {
        self.inner.drop_collection(&name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.list_collections())
    }

    fn collection_exists(&self, name: String) -> PyResult<bool> {
        Ok(self.inner.collection_exists(&name))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CollectionPy {
    inner: std::sync::Arc<openvec_core::Collection>,
}

#[pymethods]
impl CollectionPy {
    #[pyo3(signature = (doc_id, vector, payload=None))]
    fn insert(&self, doc_id: String, vector: Vec<f32>, payload: Option<pyo3::Bound<'_, PyDict>>) -> PyResult<String> {
        let mut doc = Document::new(DocumentId::from(doc_id), vector);
        if let Some(p_dict) = payload {
            let mut rust_payload = HashMap::new();
            for (key, val) in p_dict.iter() {
                let key_str = key.extract::<String>()?;
                let val_scalar = py_value_to_scalar(&val)?;
                rust_payload.insert(key_str, val_scalar);
            }
            doc.payload = rust_payload;
        }
        let inserted_id = self.inner.insert(doc)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(inserted_id.to_string())
    }

    fn delete(&self, doc_id: String) -> PyResult<bool> {
        self.inner.delete(&DocumentId::from(doc_id))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (vector, limit, ef=None, hybrid=None, vector_weight=1.0, text_weight=1.0))]
    fn search(
        &self,
        vector: Vec<f32>,
        limit: usize,
        ef: Option<usize>,
        hybrid: Option<String>,
        vector_weight: f32,
        text_weight: f32,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let mut req = SearchRequest::new(vector, limit);
        if let Some(ef_val) = ef {
            req = req.with_ef(ef_val);
        }
        if let Some(hybrid_query) = hybrid {
            req = req.with_hybrid_query(hybrid_query);
            req = req.with_weights(vector_weight, text_weight);
        }

        let results = self.inner.search(&req)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let list = PyList::empty(py);
        for res in results {
            let dict = PyDict::new(py);
            dict.set_item("id", res.id.to_string())?;
            dict.set_item("score", res.score)?;
            let payload_dict = PyDict::new(py);
            if let Some(ref p) = res.payload {
                for (k, v) in p {
                    payload_dict.set_item(k, scalar_to_py(v, py)?)?;
                }
            }
            dict.set_item("payload", payload_dict)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    fn doc_count(&self) -> PyResult<usize> {
        Ok(self.inner.doc_count())
    }

    fn index_type(&self) -> PyResult<String> {
        let index_types = self.inner.index_types();
        Ok(index_types.get("default").copied().unwrap_or("flat").to_string())
    }
}

fn py_value_to_scalar(val: &pyo3::Bound<'_, pyo3::PyAny>) -> PyResult<ScalarValue> {
    if val.is_none() {
        Ok(ScalarValue::Null)
    } else if let Ok(b) = val.extract::<bool>() {
        Ok(ScalarValue::Bool(b))
    } else if let Ok(i) = val.extract::<i64>() {
        Ok(ScalarValue::Int(i))
    } else if let Ok(f) = val.extract::<f64>() {
        Ok(ScalarValue::Float(f))
    } else if let Ok(s) = val.extract::<String>() {
        Ok(ScalarValue::Text(s))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Unsupported payload value type"))
    }
}

#[allow(deprecated)]
fn scalar_to_py(val: &ScalarValue, py: Python<'_>) -> PyResult<PyObject> {
    match val {
        ScalarValue::Null => Ok(py.None()),
        ScalarValue::Bool(b) => Ok(b.to_object(py)),
        ScalarValue::Int(i) => Ok(i.to_object(py)),
        ScalarValue::Float(f) => Ok(f.to_object(py)),
        ScalarValue::Text(s) => Ok(s.to_object(py)),
    }
}

#[pymodule]
fn openvec_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OpenVecPy>()?;
    m.add_class::<CollectionPy>()?;
    Ok(())
}
