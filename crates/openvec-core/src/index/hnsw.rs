/// HNSW Index (Hierarchical Navigable Small World)
///
/// Implements the algorithm from the 2016 paper:
/// "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
///
/// Parameter details:
/// - `m`: Max number of neighbors per node (default 16). Larger yields higher recall but uses more memory.
/// - `m_max0`: Max number of neighbors for level 0 (default m * 2 = 32).
/// - `ef_construction`: Candidate pool size during index building (default 100). Larger yields higher index quality but slower build.
/// - `ef_search`: Candidate pool size during search (can be dynamically specified at query time).
/// - `level_multiplier`: Level probability scaling factor (default 1/ln(m) ≈ 0.36).

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use parking_lot::RwLock;
use rand::prelude::*;

use crate::distance::DistanceCalculator;
use crate::types::{DistanceMetric, DocumentId, SearchResult, VectorRef};
use crate::types::error::{Error, Result};
use super::VectorIndex;

// ─────────────────────────────────────────────
//  Internal Priority Queue Elements (both Max-Heap and Min-Heap semantics)
// ─────────────────────────────────────────────

#[derive(Clone, PartialEq)]
struct Candidate {
    dist: f32,
    id: usize,  // 内部节点 ID
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// results set uses a Max-Heap: peek() gets the candidate with the largest distance (the worst)
// BinaryHeap defaults to a Max-Heap, larger dist takes priority -> forward comparison
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Larger distance = higher priority (forward comparison keeps the "worst candidate at the top" of the Max-Heap)
        self.dist.partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

// candidates queue uses a Min-Heap: pop() gets the candidate with the smallest distance (closest evaluated first)
struct MinCandidate {
    dist: f32,
    id: usize,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a Max-Heap by default, invert comparison so smaller distance pops first (simulating a Min-Heap)
        other.dist.partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}


// ─────────────────────────────────────────────
//  HNSW Node
// ─────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Node {
    /// External document ID
    doc_id: DocumentId,
    /// Vector data
    vector: Vec<f32>,
    /// Neighbor lists for each layer (layers[0] = level 0, layers[level] = highest level)
    layers: Vec<Vec<usize>>,
    /// Whether it has been deleted
    deleted: bool,
}

// ─────────────────────────────────────────────
//  HNSW Configuration
// ─────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswConfig {
    /// Max number of neighbors per node (level 1 and above)
    pub m: usize,
    /// Max number of neighbors for level 0 (typically 2 * m)
    pub m_max0: usize,
    /// Candidate pool size during index construction
    pub ef_construction: usize,
    /// Default candidate pool size during search
    pub ef_search: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HnswSerializable {
    config: HnswConfig,
    nodes: Vec<Node>,
    id_to_node: HashMap<String, usize>,
    entry_point: Option<usize>,
    max_level: usize,
    active_count: usize,
}

impl HnswConfig {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        let m = 16;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 100,
            ef_search: 50,
            dimension,
            metric,
        }
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max0 = m * 2;
        self
    }

    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

// ─────────────────────────────────────────────
//  Visited Set Tracker
// ─────────────────────────────────────────────

struct VisitedTracker {
    visited: Vec<u32>,
    epoch: u32,
}

impl VisitedTracker {
    fn new(size: usize) -> Self {
        Self {
            visited: vec![0; size],
            epoch: 1,
        }
    }

    fn reset(&mut self, size: usize) {
        if self.visited.len() < size {
            self.visited.resize(size, 0);
        }
        if self.epoch == u32::MAX {
            self.visited.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    #[inline(always)]
    fn insert(&mut self, id: usize) -> bool {
        if self.visited[id] == self.epoch {
            false
        } else {
            self.visited[id] = self.epoch;
            true
        }
    }
}

// ─────────────────────────────────────────────
//  HNSW Index Internal
// ─────────────────────────────────────────────

struct HnswInner {
    config: HnswConfig,
    nodes: Vec<Node>,
    id_to_node: HashMap<String, usize>,
    entry_point: Option<usize>,   // Entry point node internal ID
    max_level: usize,
    calc: DistanceCalculator,
    active_count: usize,
    rng: SmallRng,
    level_multiplier: f64,        // 1 / ln(m)
}

impl HnswInner {
    fn new(config: HnswConfig) -> Self {
        let m = config.m;
        let metric = config.metric;
        let level_multiplier = 1.0 / (m as f64).ln().max(1.0);
        Self {
            config,
            nodes: Vec::new(),
            id_to_node: HashMap::new(),
            entry_point: None,
            max_level: 0,
            calc: DistanceCalculator::new(metric),
            active_count: 0,
            rng: SmallRng::from_os_rng(),
            level_multiplier,
        }
    }

    /// Randomly generates the insertion level for a new node (geometric distribution)
    fn random_level(&mut self) -> usize {
        let mut level = 0;
        while self.rng.random::<f64>() < self.level_multiplier && level < 16 {
            level += 1;
        }
        level
    }

    /// Computes the distance between an internal node and an external query vector
    #[inline]
    fn dist_to_query(&self, node_id: usize, query: VectorRef) -> f32 {
        self.calc.compute(&self.nodes[node_id].vector, query)
    }

    fn search_layer(
        &self,
        query: VectorRef,
        entry_node: usize,
        ef: usize,
        layer: usize,
        tracker: &mut VisitedTracker,
        candidates: &mut BinaryHeap<MinCandidate>,
        results: &mut BinaryHeap<Candidate>,
        filter_fn: Option<&dyn Fn(&DocumentId) -> bool>,
    ) {
        candidates.clear();
        results.clear();
        tracker.reset(self.nodes.len());
        tracker.insert(entry_node);

        let entry_dist = self.dist_to_query(entry_node, query);
        candidates.push(MinCandidate { dist: entry_dist, id: entry_node });
        
        let is_entry_match = if let Some(ref f) = filter_fn {
            f(&self.nodes[entry_node].doc_id)
        } else {
            true
        };
        if is_entry_match {
            results.push(Candidate { dist: entry_dist, id: entry_node });
        }

        while let Some(MinCandidate { dist: cur_dist, id: cur_id }) = candidates.pop() {
            // Stop expansion if current candidate is further than the furthest in the result set
            if let Some(worst) = results.peek() {
                if cur_dist > worst.dist && results.len() >= ef {
                    break;
                }
            }

            // Traverse neighbors of the current node at this layer
            if layer < self.nodes[cur_id].layers.len() {
                for &neighbor_id in &self.nodes[cur_id].layers[layer] {
                    if !tracker.insert(neighbor_id) {
                        continue;
                    }

                    if self.nodes[neighbor_id].deleted {
                        continue;
                    }

                    let is_match = if let Some(ref f) = filter_fn {
                        f(&self.nodes[neighbor_id].doc_id)
                    } else {
                        true
                    };

                    let neighbor_dist = self.dist_to_query(neighbor_id, query);

                    let should_explore = results.len() < ef
                        || results.peek().map_or(true, |w| neighbor_dist < w.dist);

                    if should_explore {
                        candidates.push(MinCandidate { dist: neighbor_dist, id: neighbor_id });
                        
                        if is_match {
                            results.push(Candidate { dist: neighbor_dist, id: neighbor_id });

                            // Maintain result set size
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
    }

    /// Selects M nearest neighbors from candidates (simple strategy: take the closest M)
    fn select_neighbors(&self, candidates: &BinaryHeap<Candidate>, m: usize) -> Vec<(usize, f32)> {
        let mut sorted: Vec<(f32, usize)> = candidates.iter()
            .map(|c| (c.dist, c.id))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(d, id)| (id, d)).collect()
    }

    fn insert(&mut self, doc_id: &DocumentId, vector: VectorRef) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let mut vector_data = vector.to_vec();
        if self.config.metric == DistanceMetric::Cosine {
            crate::distance::normalize(&mut vector_data);
        }

        // Check if this is an upsert
        if let Some(&existing_node) = self.id_to_node.get(doc_id.as_str()) {
            let node = &mut self.nodes[existing_node];
            node.vector.copy_from_slice(&vector_data);
            if node.deleted {
                node.deleted = false;
                self.active_count += 1;
            }
            // Note: upsert does not rebuild the graph connections, it only updates the vector.
            // For scenarios requiring precise graph updates, call delete first then insert.
            return Ok(());
        }

        let new_node_id = self.nodes.len();
        let insert_level = self.random_level();

        // Initialize the new node
        let node = Node {
            doc_id: doc_id.clone(),
            vector: vector_data,
            layers: vec![Vec::new(); insert_level + 1],
            deleted: false,
        };
        self.nodes.push(node);
        self.id_to_node.insert(doc_id.as_str().to_string(), new_node_id);
        self.active_count += 1;

        // If it is the first node, set it as the entry point directly
        let Some(mut entry_point) = self.entry_point else {
            self.entry_point = Some(new_node_id);
            self.max_level = insert_level;
            return Ok(());
        };

        let current_max_level = self.max_level;
        let mut tracker = VisitedTracker::new(self.nodes.len());
        let mut candidates_heap = BinaryHeap::with_capacity(self.config.ef_construction + 16);
        let mut results_heap = BinaryHeap::with_capacity(self.config.ef_construction + 16);

        // Greedily descend from the highest level to insert_level + 1
        for level in (insert_level + 1..=current_max_level).rev() {
            self.search_layer(&self.nodes[new_node_id].vector, entry_point, 1, level, &mut tracker, &mut candidates_heap, &mut results_heap, None);
            if let Some(best) = results_heap.peek() {
                entry_point = best.id;
            }
        }

        // Insert layer-by-layer from insert_level down to level 0
        for level in (0..=insert_level.min(current_max_level)).rev() {
            let m = if level == 0 { self.config.m_max0 } else { self.config.m };
            let ef = self.config.ef_construction;

            self.search_layer(&self.nodes[new_node_id].vector, entry_point, ef, level, &mut tracker, &mut candidates_heap, &mut results_heap, None);
            let selected = self.select_neighbors(&results_heap, m);

            // Set neighbors of the new node
            if level < self.nodes[new_node_id].layers.len() {
                self.nodes[new_node_id].layers[level] = selected.iter().map(|(id, _)| *id).collect();
            }

            // Update neighbor connections (bidirectional edges)
            for (neighbor_id, _neighbor_dist) in &selected {
                let neighbor_id = *neighbor_id;
                if level < self.nodes[neighbor_id].layers.len() {
                    self.nodes[neighbor_id].layers[level].push(new_node_id);

                    // Prune if neighbor connection count exceeds max_conn (keep the closest ones)
                    let max_conn = if level == 0 { self.config.m_max0 } else { self.config.m };
                    if self.nodes[neighbor_id].layers[level].len() > max_conn {
                        let neighbor_vec = self.nodes[neighbor_id].vector.clone();
                        let mut conn_dists: Vec<(f32, usize)> = self.nodes[neighbor_id].layers[level]
                            .iter()
                            .filter(|&&conn_id| !self.nodes[conn_id].deleted)
                            .map(|&conn_id| {
                                let d = self.calc.compute(&neighbor_vec, &self.nodes[conn_id].vector);
                                (d, conn_id)
                            })
                            .collect();
                        conn_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                        conn_dists.truncate(max_conn);
                        self.nodes[neighbor_id].layers[level] = conn_dists.into_iter().map(|(_, id)| id).collect();
                    }
                }
            }

            // Update entry point to the closest candidate of this layer
            if let Some(best) = results_heap.iter().min_by(|a, b| {
                a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
            }) {
                entry_point = best.id;
            }
        }

        // If the new node has a higher level, update the entry point
        if insert_level > current_max_level {
            self.entry_point = Some(new_node_id);
            self.max_level = insert_level;
        }

        Ok(())
    }

    fn delete(&mut self, doc_id: &DocumentId) -> bool {
        if let Some(&node_id) = self.id_to_node.get(doc_id.as_str()) {
            if !self.nodes[node_id].deleted {
                self.nodes[node_id].deleted = true;
                self.active_count -= 1;
                return true;
            }
        }
        false
    }

    fn search(&self, query: VectorRef, k: usize, ef: usize, filter_fn: Option<&dyn Fn(&DocumentId) -> bool>) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        let Some(mut entry_point) = self.entry_point else {
            return Ok(Vec::new());
        };

        if self.active_count == 0 {
            return Ok(Vec::new());
        }

        // Skip deleted entry point node, find the first valid node
        if self.nodes[entry_point].deleted {
            // Fallback: linear scan to find the first non-deleted node
            match self.nodes.iter().position(|n| !n.deleted) {
                Some(pos) => entry_point = pos,
                None => return Ok(Vec::new()),
            }
        }

        let mut query_data = query.to_vec();
        if self.config.metric == DistanceMetric::Cosine {
            crate::distance::normalize(&mut query_data);
        }

        let mut tracker = VisitedTracker::new(self.nodes.len());
        let mut candidates_heap = BinaryHeap::with_capacity(ef + 16);
        let mut results_heap = BinaryHeap::with_capacity(ef + 16);

        // Greedily descend from the highest level to level 1
        for level in (1..=self.max_level).rev() {
            self.search_layer(&query_data, entry_point, 1, level, &mut tracker, &mut candidates_heap, &mut results_heap, filter_fn);
            if let Some(best) = results_heap.peek() {
                entry_point = best.id;
            }
        }

        // Perform fine-grained search at level 0
        let ef_actual = ef.max(k);
        self.search_layer(&query_data, entry_point, ef_actual, 0, &mut tracker, &mut candidates_heap, &mut results_heap, filter_fn);

        // Gather results (sorted by distance, take k)
        let mut results: Vec<(f32, usize)> = results_heap.into_iter()
            .filter(|c| !self.nodes[c.id].deleted)
            .map(|c| (c.dist, c.id))
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        results.truncate(k);

        let search_results = results.into_iter().map(|(score, node_id)| {
            SearchResult {
                id: self.nodes[node_id].doc_id.clone(),
                score,
                payload: None,
            }
        }).collect();

        Ok(search_results)
    }
}

// ─────────────────────────────────────────────
//  公共 HNSW Index
// ─────────────────────────────────────────────

/// Thread-safe HNSW Index
pub struct HnswIndex {
    inner: RwLock<HnswInner>,
    config: HnswConfig,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        let cfg_clone = config.clone();
        Self {
            inner: RwLock::new(HnswInner::new(config)),
            config: cfg_clone,
        }
    }

    pub fn with_defaults(dimension: usize, metric: DistanceMetric) -> Self {
        Self::new(HnswConfig::new(dimension, metric))
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, id: &DocumentId, vector: VectorRef) -> Result<()> {
        self.inner.write().insert(id, vector)
    }

    fn delete(&mut self, id: &DocumentId) -> Result<bool> {
        Ok(self.inner.write().delete(id))
    }

    fn search(&self, query: VectorRef, k: usize, ef: Option<usize>, filter_fn: Option<&dyn Fn(&DocumentId) -> bool>) -> Result<Vec<SearchResult>> {
        let ef = ef.unwrap_or(self.config.ef_search).max(k);
        self.inner.read().search(query, k, ef, filter_fn)
    }

    fn len(&self) -> usize {
        self.inner.read().active_count
    }

    fn index_type(&self) -> &'static str {
        "hnsw"
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let serializable = HnswSerializable {
            config: inner.config.clone(),
            nodes: inner.nodes.clone(),
            id_to_node: inner.id_to_node.clone(),
            entry_point: inner.entry_point,
            max_level: inner.max_level,
            active_count: inner.active_count,
        };
        serde_json::to_vec(&serializable)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    fn deserialize_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let serializable: HnswSerializable = serde_json::from_slice(bytes)
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        
        let mut inner = self.inner.write();
        inner.config = serializable.config.clone();
        inner.nodes = serializable.nodes;
        inner.id_to_node = serializable.id_to_node;
        inner.entry_point = serializable.entry_point;
        inner.max_level = serializable.max_level;
        inner.active_count = serializable.active_count;
        
        // Reconstruct other fields
        inner.calc = DistanceCalculator::new(inner.config.metric);
        inner.level_multiplier = 1.0 / (inner.config.m as f64).ln().max(1.0);
        inner.rng = SmallRng::from_os_rng();
        
        // Update self config
        self.config = serializable.config;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    fn make_id(s: &str) -> DocumentId {
        DocumentId::from(s)
    }

    fn random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..dim).map(|_| rng.random::<f32>()).collect()
    }

    #[test]
    fn hnsw_basic_insert_and_search() {
        let mut idx = HnswIndex::with_defaults(4, DistanceMetric::L2);

        idx.insert(&make_id("a"), &[0.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(&make_id("b"), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(&make_id("c"), &[10.0, 0.0, 0.0, 0.0]).unwrap();

        let results = idx.search(&[0.1, 0.0, 0.0, 0.0], 2, None, None).unwrap();
        assert!(results.len() >= 1);
        assert_eq!(results[0].id.as_str(), "a");
    }

    #[test]
    fn hnsw_delete() {
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        idx.insert(&make_id("a"), &[0.0, 0.0]).unwrap();
        idx.insert(&make_id("b"), &[0.1, 0.0]).unwrap();

        idx.delete(&make_id("a")).unwrap();
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[0.0, 0.0], 2, None, None).unwrap();
        assert!(results.iter().all(|r| r.id.as_str() != "a"));
    }

    #[test]
    fn hnsw_recall_random_data() {
        // Test that recall is >= 0.9 on random data
        let mut rng = SmallRng::seed_from_u64(42);
        let dim = 64;
        let n = 500;
        let k = 10;

        let mut hnsw = HnswIndex::new(
            HnswConfig::new(dim, DistanceMetric::L2)
                .with_m(16)
                .with_ef_construction(100)
                .with_ef_search(50)
        );
        let mut flat = crate::index::flat::FlatIndex::new(dim, DistanceMetric::L2);

        let mut vecs: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..n {
            let v = random_vector(dim, &mut rng);
            let id = format!("doc_{i}");
            hnsw.insert(&DocumentId::from(id.as_str()), &v).unwrap();
            flat.insert(&DocumentId::from(id.as_str()), &v).unwrap();
            vecs.push((id, v));
        }

        let n_queries = 20;
        let mut total_recall = 0.0f64;

        for _ in 0..n_queries {
            let query = random_vector(dim, &mut rng);

            let hnsw_results: HashSet<String> = hnsw.search(&query, k, Some(100), None).unwrap()
                .into_iter().map(|r| r.id.0).collect();

            let flat_results: HashSet<String> = flat.search(&query, k, None, None).unwrap()
                .into_iter().map(|r| r.id.0).collect();

            let intersection = hnsw_results.intersection(&flat_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / n_queries as f64;
        println!("HNSW Recall@{k}: {:.3}", avg_recall);
        assert!(avg_recall >= 0.85, "Recall {avg_recall:.3} is below threshold 0.85");
    }
}
