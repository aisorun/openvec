use std::collections::HashMap;
use std::cmp::Ordering;
use parking_lot::RwLock;

use crate::distance::DistanceCalculator;
use crate::types::{DistanceMetric, DocumentId, SearchResult, VectorRef};
use crate::types::error::{Error, Result};
use super::VectorIndex;

// ─────────────────────────────────────────────
//  Lloyd's K-Means Clustering for Centroids
// ─────────────────────────────────────────────

/// Runs K-Means clustering on the given vectors to find `k` centroids
pub fn train_kmeans(vectors: &[&[f32]], k: usize, max_iterations: usize, calc: &DistanceCalculator) -> Vec<Vec<f32>> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let dim = vectors[0].len();

    // 1. Initial centroids: Deterministic Farthest Point Clustering initialization (SOTA deterministic K-Means++ variant)
    let mut centroids = Vec::with_capacity(k);
    centroids.push(vectors[0].to_vec());

    while centroids.len() < k {
        let mut farthest_idx = 0;
        let mut max_min_dist = -1.0f32;

        for (i, vec) in vectors.iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            for centroid in &centroids {
                let dist = calc.compute(vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                farthest_idx = i;
            }
        }
        centroids.push(vectors[farthest_idx].to_vec());
    }

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iterations {
        let mut changed = false;

        // Assignment step: Assign each vector to the nearest centroid
        for (i, vec) in vectors.iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            let mut best_centroid = 0;

            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist = calc.compute(vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_centroid = c_idx;
                }
            }

            if assignments[i] != best_centroid {
                assignments[i] = best_centroid;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: Recompute centroids as averages of assigned vectors
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &c_idx) in assignments.iter().enumerate() {
            let vec = vectors[i];
            for d in 0..dim {
                new_centroids[c_idx][d] += vec[d];
            }
            counts[c_idx] += 1;
        }

        for c_idx in 0..k {
            if counts[c_idx] > 0 {
                let count = counts[c_idx] as f32;
                for d in 0..dim {
                    new_centroids[c_idx][d] /= count;
                }
                centroids[c_idx] = new_centroids[c_idx].clone();
            }
        }
    }

    centroids
}

// ─────────────────────────────────────────────
//  Dimensional 8-bit Scalar Quantization (SQ8)
// ─────────────────────────────────────────────

/// Precomputed Look-up Table for 8-bit Scalar Quantized Asymmetric Distance Computation
pub struct Sq8Lut {
    // 2D table: dim x 256
    pub table: Vec<[f32; 256]>,
}

impl Sq8Lut {
    #[inline(always)]
    pub fn compute_l2(&self, qvec: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &q) in qvec.iter().enumerate() {
            sum += self.table[i][q as usize];
        }
        sum.sqrt()
    }

    #[inline(always)]
    pub fn compute_dot(&self, qvec: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &q) in qvec.iter().enumerate() {
            sum += self.table[i][q as usize];
        }
        -sum
    }

    #[inline(always)]
    pub fn compute_cosine(&self, qvec: &[u8], norm_q: f32, norm_x: f32) -> f32 {
        let mut dot = 0.0f32;
        for (i, &q) in qvec.iter().enumerate() {
            dot += self.table[i][q as usize];
        }
        let denom = norm_q * norm_x;
        if denom < 1e-6 {
            1.0
        } else {
            let similarity = (dot / denom).clamp(-1.0, 1.0);
            1.0 - similarity
        }
    }
}

/// High-precision dimensional scalar quantizer (min/max tracked per dimension)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalarQuantizer {
    pub mins: Vec<f32>,
    pub maxes: Vec<f32>,
}

impl ScalarQuantizer {
    pub fn train(vectors: &[&[f32]], dimension: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dimension];
        let mut maxes = vec![f32::NEG_INFINITY; dimension];

        for vec in vectors {
            for i in 0..dimension {
                mins[i] = mins[i].min(vec[i]);
                maxes[i] = maxes[i].max(vec[i]);
            }
        }

        Self { mins, maxes }
    }

    /// Builds a Look-Up Table (LUT) for query asymmetric distance calculation
    pub fn build_lut(&self, query: &[f32], metric: DistanceMetric) -> Sq8Lut {
        let dim = query.len();
        let mut table = vec![[0.0f32; 256]; dim];
        for i in 0..dim {
            let min = self.mins[i];
            let max = self.maxes[i];
            let range = max - min;
            let q_val = query[i];
            for q in 0..256 {
                let dequant_val = min + (q as f32 / 255.0) * range;
                table[i][q] = match metric {
                    DistanceMetric::L2 => {
                        let diff = q_val - dequant_val;
                        diff * diff
                    }
                    DistanceMetric::Cosine | DistanceMetric::DotProduct => {
                        q_val * dequant_val
                    }
                };
            }
        }
        Sq8Lut { table }
    }

    /// Compresses f32 vector into u8 bytes
    pub fn quantize(&self, vec: &[f32]) -> Vec<u8> {
        vec.iter()
            .zip(self.mins.iter().zip(self.maxes.iter()))
            .map(|(&x, (&min, &max))| {
                let range = max - min;
                if range < 1e-6 {
                    0
                } else {
                    let q = ((x - min) / range * 255.0).round();
                    q.clamp(0.0, 255.0) as u8
                }
            })
            .collect()
    }

    /// Restores quantized bytes back to f32 vector
    pub fn dequantize(&self, qvec: &[u8]) -> Vec<f32> {
        qvec.iter()
            .zip(self.mins.iter().zip(self.maxes.iter()))
            .map(|(&q, (&min, &max))| min + (q as f32 / 255.0) * (max - min))
            .collect()
    }
}

// ─────────────────────────────────────────────
//  IVF-SQ8 Index Structure
// ─────────────────────────────────────────────

/// Internal IVF-SQ8 State
struct IvfSq8Inner {
    dimension: usize,
    calc: DistanceCalculator,
    quantizer: Option<ScalarQuantizer>,
    centroids: Vec<Vec<f32>>,
    // postings: centroids_idx -> Vec<(doc_id, quantized_vector, vector_norm, deleted)>
    postings: Vec<Vec<(DocumentId, Vec<u8>, f32, bool)>>,
    // Cold start cache for training (stores raw vectors until training threshold)
    training_cache: Vec<(DocumentId, Vec<f32>)>,
    training_threshold: usize,
    n_centroids: usize,
    n_probe: usize,
    active_count: usize,
    // doc_id -> (centroid_idx, postings_list_idx) for fast soft delete/resurrection
    doc_to_posting: HashMap<String, (usize, usize)>,
}

impl IvfSq8Inner {
    fn new(dimension: usize, metric: DistanceMetric) -> Self {
        let n_centroids = 16; // default centroids
        let n_probe = 4;     // default probe
        Self {
            dimension,
            calc: DistanceCalculator::new(metric),
            quantizer: None,
            centroids: Vec::new(),
            postings: vec![Vec::new(); n_centroids],
            training_cache: Vec::new(),
            training_threshold: 64, // Train once we hit 64 vectors (makes testing easy)
            n_centroids,
            n_probe,
            active_count: 0,
            doc_to_posting: HashMap::new(),
        }
    }

    fn train(&mut self) -> Result<()> {
        if self.training_cache.len() < self.n_centroids {
            return Ok(());
        }

        let raw_vectors: Vec<&[f32]> = self.training_cache.iter().map(|(_, v)| v.as_slice()).collect();

        // 1. Train centroids
        let centroids = train_kmeans(&raw_vectors, self.n_centroids, 15, &self.calc);
        self.centroids = centroids;

        // 2. Train quantizer mins/maxes
        let quantizer = ScalarQuantizer::train(&raw_vectors, self.dimension);
        self.quantizer = Some(quantizer.clone());

        // 3. Move items from cache to postings
        let cache_docs = std::mem::take(&mut self.training_cache);
        for (doc_id, vec) in cache_docs {
            self.insert_quantized(&doc_id, &vec)?;
        }

        Ok(())
    }

    fn insert_quantized(&mut self, doc_id: &DocumentId, vector: &[f32]) -> Result<()> {
        let quantizer = self.quantizer.as_ref().unwrap();

        // Find nearest centroid
        let mut min_dist = f32::INFINITY;
        let mut best_centroid = 0;

        for (c_idx, centroid) in self.centroids.iter().enumerate() {
            let dist = self.calc.compute(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_centroid = c_idx;
            }
        }

        let qvec = quantizer.quantize(vector);
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Check for upsert/resurrection
        if let Some(&(c_idx, p_idx)) = self.doc_to_posting.get(doc_id.as_str()) {
            let entry = &mut self.postings[c_idx][p_idx];
            if entry.3 {
                entry.3 = false;
                self.active_count += 1;
            }
            entry.1 = qvec;
            entry.2 = norm;
            return Ok(());
        }

        let pos = self.postings[best_centroid].len();
        self.postings[best_centroid].push((doc_id.clone(), qvec, norm, false));
        self.doc_to_posting.insert(doc_id.as_str().to_string(), (best_centroid, pos));
        self.active_count += 1;

        Ok(())
    }

    fn insert(&mut self, doc_id: &DocumentId, vector: VectorRef) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let mut vector_data = vector.to_vec();
        if self.calc.metric() == DistanceMetric::Cosine {
            crate::distance::normalize(&mut vector_data);
        }

        // If trained, route to postings directly
        if self.quantizer.is_some() {
            return self.insert_quantized(doc_id, &vector_data);
        }

        // Otherwise, add to cold cache
        if let Some(pos) = self.training_cache.iter().position(|(id, _)| id == doc_id) {
            self.training_cache[pos].1 = vector_data;
            return Ok(());
        }

        self.training_cache.push((doc_id.clone(), vector_data));
        self.active_count += 1;

        // Check if we hit the training threshold
        if self.training_cache.len() >= self.training_threshold {
            self.train()?;
        }

        Ok(())
    }

    fn delete(&mut self, doc_id: &DocumentId) -> bool {
        // Check postings
        if let Some(&(c_idx, p_idx)) = self.doc_to_posting.get(doc_id.as_str()) {
            let entry = &mut self.postings[c_idx][p_idx];
            if !entry.3 {
                entry.3 = true;
                self.active_count = self.active_count.saturating_sub(1);
                return true;
            }
            return false;
        }

        // Check training cache
        if let Some(pos) = self.training_cache.iter().position(|(id, _)| id == doc_id) {
            self.training_cache.remove(pos);
            self.active_count = self.active_count.saturating_sub(1);
            return true;
        }

        false
    }

    fn search(&self, query: VectorRef, k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        if self.active_count == 0 {
            return Ok(Vec::new());
        }

        let mut query_data = query.to_vec();
        if self.calc.metric() == DistanceMetric::Cosine {
            crate::distance::normalize(&mut query_data);
        }

        // 1. Cold start cache fallback search (Exact flat search)
        if self.quantizer.is_none() {
            let mut distances: Vec<(f32, &DocumentId)> = self.training_cache.iter()
                .map(|(id, vec)| {
                    let d = self.calc.compute(&query_data, vec);
                    (d, id)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            distances.truncate(k);

            let results = distances.into_iter().map(|(score, id)| SearchResult {
                id: id.clone(),
                score,
                payload: None,
            }).collect();

            return Ok(results);
        }

        let quantizer = self.quantizer.as_ref().unwrap();

        // 2. Build the LUT for the query vector (SOTA Asymmetric Distance Computation)
        let lut = quantizer.build_lut(&query_data, self.calc.metric());
        let norm_q = if self.calc.metric() == DistanceMetric::Cosine {
            query_data.iter().map(|&x| x * x).sum::<f32>().sqrt()
        } else {
            0.0f32
        };

        // Find n_probe nearest centroids to query vector
        let mut centroid_dists: Vec<(f32, usize)> = self.centroids.iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let d = self.calc.compute(&query_data, centroid);
                (d, idx)
            })
            .collect();

        centroid_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let probe_limit = self.n_probe.min(centroid_dists.len());

        let mut candidates = Vec::new();

        // Scan postings lists of top centroids
        for i in 0..probe_limit {
            let c_idx = centroid_dists[i].1;
            for (doc_id, qvec, norm_x, deleted) in &self.postings[c_idx] {
                if *deleted {
                    continue;
                }

                // SOTA Asymmetric Distance Computation via LUT lookups (zero heap allocations or floats math)
                let score = match self.calc.metric() {
                    DistanceMetric::L2 => lut.compute_l2(qvec),
                    DistanceMetric::DotProduct => lut.compute_dot(qvec),
                    DistanceMetric::Cosine => lut.compute_cosine(qvec, norm_q, *norm_x),
                };
                candidates.push((score, doc_id));
            }
        }

        // Sort candidates
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        candidates.truncate(k);

        let search_results = candidates.into_iter().map(|(score, id)| SearchResult {
            id: id.clone(),
            score,
            payload: None,
        }).collect();

        Ok(search_results)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IvfSq8Serializable {
    dimension: usize,
    metric: DistanceMetric,
    quantizer: Option<ScalarQuantizer>,
    centroids: Vec<Vec<f32>>,
    postings: Vec<Vec<(DocumentId, Vec<u8>, f32, bool)>>,
    training_cache: Vec<(DocumentId, Vec<f32>)>,
    training_threshold: usize,
    n_centroids: usize,
    n_probe: usize,
    active_count: usize,
    doc_to_posting: HashMap<String, (usize, usize)>,
}

/// Thread-safe IVF-SQ8 Index
pub struct IvfSq8Index {
    inner: RwLock<IvfSq8Inner>,
    dimension: usize,
}

impl IvfSq8Index {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            inner: RwLock::new(IvfSq8Inner::new(dimension, metric)),
            dimension,
        }
    }
}

impl VectorIndex for IvfSq8Index {
    fn insert(&mut self, id: &DocumentId, vector: VectorRef) -> Result<()> {
        self.inner.write().insert(id, vector)
    }

    fn delete(&mut self, id: &DocumentId) -> Result<bool> {
        Ok(self.inner.write().delete(id))
    }

    fn search(&self, query: VectorRef, k: usize, _ef: Option<usize>) -> Result<Vec<SearchResult>> {
        self.inner.read().search(query, k)
    }

    fn len(&self) -> usize {
        self.inner.read().active_count
    }

    fn index_type(&self) -> &'static str {
        "ivf_sq8"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let serializable = IvfSq8Serializable {
            dimension: inner.dimension,
            metric: inner.calc.metric(),
            quantizer: inner.quantizer.clone(),
            centroids: inner.centroids.clone(),
            postings: inner.postings.clone(),
            training_cache: inner.training_cache.clone(),
            training_threshold: inner.training_threshold,
            n_centroids: inner.n_centroids,
            n_probe: inner.n_probe,
            active_count: inner.active_count,
            doc_to_posting: inner.doc_to_posting.clone(),
        };
        serde_json::to_vec(&serializable)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    fn deserialize_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let serializable: IvfSq8Serializable = serde_json::from_slice(bytes)
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        
        let mut inner = self.inner.write();
        inner.dimension = serializable.dimension;
        inner.calc = DistanceCalculator::new(serializable.metric);
        inner.quantizer = serializable.quantizer;
        inner.centroids = serializable.centroids;
        inner.postings = serializable.postings;
        inner.training_cache = serializable.training_cache;
        inner.training_threshold = serializable.training_threshold;
        inner.n_centroids = serializable.n_centroids;
        inner.n_probe = serializable.n_probe;
        inner.active_count = serializable.active_count;
        inner.doc_to_posting = serializable.doc_to_posting;
        
        self.dimension = serializable.dimension;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(s: &str) -> DocumentId {
        DocumentId::from(s)
    }

    #[test]
    fn test_sq8_quantizer_reconstruction() {
        let vectors = vec![
            vec![0.0, 1.0, 10.0],
            vec![2.0, -1.0, 5.0],
            vec![-1.0, 2.0, -2.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let quantizer = ScalarQuantizer::train(&refs, 3);

        // Verify min/max dimensions
        assert_eq!(quantizer.mins, vec![-1.0, -1.0, -2.0]);
        assert_eq!(quantizer.maxes, vec![2.0, 2.0, 10.0]);

        // Quantize and dequantize
        let test_vec = vec![1.0, 0.0, 4.0];
        let q = quantizer.quantize(&test_vec);
        let dq = quantizer.dequantize(&q);

        // SQ8 reconstruction error should be very small
        for i in 0..3 {
            assert!((test_vec[i] - dq[i]).abs() < 0.1, "SQ8 dimension {i} error too high");
        }
    }

    #[test]
    fn test_ivf_sq8_cold_start_and_transition() {
        // Initialize an IVF-SQ8 index with dimension 2
        let mut idx = IvfSq8Index::new(2, DistanceMetric::L2);

        // Before training threshold (64 docs), it works in fallback cache mode
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        idx.insert(&make_id("b"), &[1.0, 0.0]).unwrap();
        idx.insert(&make_id("c"), &[2.0, 0.0]).unwrap();

        assert_eq!(idx.len(), 3);
        assert_eq!(idx.inner.read().quantizer.is_none(), true); // still in fallback

        let res = idx.search(&[0.1, 0.0], 2, None).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id.as_str(), "a");
        assert_eq!(res[1].id.as_str(), "b");

        // Insert up to 64 docs to trigger automatic K-Means & SQ8 Quantization training
        for i in 0..64 {
            let val = i as f32 * 0.1;
            idx.insert(&make_id(&format!("doc_{i}")), &[val, val]).unwrap();
        }

        // Verify that the index was successfully trained and transitioned
        assert_eq!(idx.inner.read().quantizer.is_some(), true);
        assert_eq!(idx.inner.read().centroids.len(), 16);

        // Verify searches still yield top ranks
        let res = idx.search(&[0.11, 0.11], 2, None).unwrap();
        assert_eq!(res[0].id.as_str(), "doc_1");
    }

    #[test]
    fn test_sq8_lut_distance_accuracy() {
        // Setup simple quantizer for 3D vectors
        let vectors = vec![
            vec![0.0, 1.0, 10.0],
            vec![2.0, -1.0, 5.0],
            vec![-1.0, 2.0, -2.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let quantizer = ScalarQuantizer::train(&refs, 3);

        let query = vec![1.0, 0.0, 4.0];

        // L2 metric LUT check
        let lut_l2 = quantizer.build_lut(&query, DistanceMetric::L2);
        let test_vec = vec![1.5, 0.5, 3.5];
        let q = quantizer.quantize(&test_vec);
        let dq = quantizer.dequantize(&q);

        // Standard dequantized L2 distance
        let standard_l2 = crate::distance::raw::l2_distance(&query, &dq);
        // LUT-based L2 distance
        let lut_l2_dist = lut_l2.compute_l2(&q);

        assert!((standard_l2 - lut_l2_dist).abs() < 1e-4, "L2 distance via LUT deviates from standard");

        // Dot product metric LUT check
        let lut_dot = quantizer.build_lut(&query, DistanceMetric::DotProduct);
        let standard_dot = crate::distance::raw::dot_distance(&query, &dq);
        let lut_dot_dist = lut_dot.compute_dot(&q);
        assert!((standard_dot - lut_dot_dist).abs() < 1e-4, "DotProduct via LUT deviates from standard");

        // Cosine metric LUT check
        let norm_q = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_x = dq.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let lut_cos = quantizer.build_lut(&query, DistanceMetric::Cosine);
        let standard_cos = crate::distance::raw::cosine_distance(&query, &dq);
        let lut_cos_dist = lut_cos.compute_cosine(&q, norm_q, norm_x);
        assert!((standard_cos - lut_cos_dist).abs() < 1e-4, "Cosine via LUT deviates from standard");
    }

    #[test]
    fn test_kmeans_fpc_centroid_dispersion() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![-5.0, -5.0],
            vec![-6.0, -6.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let calc = DistanceCalculator::new(DistanceMetric::L2);

        // Train 3 centroids using Farthest Point Clustering initialization + Lloyd's iterations
        let centroids = train_kmeans(&refs, 3, 10, &calc);
        assert_eq!(centroids.len(), 3);

        // Check that centroids correctly capture the 3 dispersed groups:
        // Group A: [0.0, 0.0] or [1.0, 0.0] -> close to [0.5, 0.0]
        // Group B: [10.0, 10.0] or [11.0, 11.0] -> close to [10.5, 10.5]
        // Group C: [-5.0, -5.0] or [-6.0, -6.0] -> close to [-5.5, -5.5]
        let mut group_a = false;
        let mut group_b = false;
        let mut group_c = false;

        for c in centroids {
            if (c[0] - 0.5).abs() < 1.0 && (c[1] - 0.0).abs() < 1.0 {
                group_a = true;
            } else if (c[0] - 10.5).abs() < 1.0 && (c[1] - 10.5).abs() < 1.0 {
                group_b = true;
            } else if (c[0] - (-5.5)).abs() < 1.0 && (c[1] - (-5.5)).abs() < 1.0 {
                group_c = true;
            }
        }

        assert!(group_a, "Centroids missed Group A");
        assert!(group_b, "Centroids missed Group B");
        assert!(group_c, "Centroids missed Group C");
    }
}
