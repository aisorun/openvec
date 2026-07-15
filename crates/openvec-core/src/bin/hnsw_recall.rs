use openvec_core::types::{DistanceMetric, DocumentId};
use openvec_core::index::{VectorIndex, flat::FlatIndex, hnsw::{HnswConfig, HnswIndex}};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::collections::HashSet;
use std::time::Instant;

fn random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.random::<f32>()).collect()
}

fn main() {
    println!("==========================================");
    println!("  OpenVec HNSW 准确率 (Recall) 与延迟评估 ");
    println!("==========================================");

    let dim = 128;
    let n_docs = 10000;
    let n_queries = 100;
    let k = 10;

    println!("正在生成测试数据...");
    println!("- 向量维度: {}", dim);
    println!("- 向量数量: {}", n_docs);
    println!("- 查询数量: {}", n_queries);
    println!("- 检索个数 K: {}", k);

    let mut rng = SmallRng::seed_from_u64(42);
    let mut vecs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_docs {
        vecs.push(random_vector(dim, &mut rng));
    }

    let mut queries: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_queries {
        queries.push(random_vector(dim, &mut rng));
    }

    println!("\n正在构建 FlatIndex (作为 Ground Truth)...");
    let mut flat = FlatIndex::new(dim, DistanceMetric::L2);
    let flat_build_start = Instant::now();
    for (i, v) in vecs.iter().enumerate() {
        let id = DocumentId::from(format!("doc_{}", i).as_str());
        flat.insert(&id, v).unwrap();
    }
    println!("FlatIndex 构建完毕，耗时: {:?}", flat_build_start.elapsed());

    println!("\n正在构建 HNSW 索引 (M=16, ef_construction=100)...");
    let mut hnsw = HnswIndex::new(
        HnswConfig::new(dim, DistanceMetric::L2)
            .with_m(16)
            .with_ef_construction(100)
    );
    let hnsw_build_start = Instant::now();
    for (i, v) in vecs.iter().enumerate() {
        let id = DocumentId::from(format!("doc_{}", i).as_str());
        hnsw.insert(&id, v).unwrap();
    }
    println!("HNSW 索引构建完毕，耗时: {:?}", hnsw_build_start.elapsed());

    // 预计算 Ground Truth
    println!("\n预计算 Ground Truth 中...");
    let mut ground_truths: Vec<HashSet<String>> = Vec::new();
    for q in &queries {
        let flat_results: HashSet<String> = flat.search(q, k, None, None).unwrap()
            .into_iter()
            .map(|r| r.id.0)
            .collect();
        ground_truths.push(flat_results);
    }
    println!("Ground Truth 计算完毕。");

    // 评测不同 ef_search 下的召回率与耗时
    let ef_searches = [10, 20, 50, 100, 150, 200];
    
    println!("\n| ef_search | 召回率 Recall@10 | 平均查询延迟 (µs) | 总查询耗时 |");
    println!("| :---: | :---: | :---: | :---: |");

    for &ef in &ef_searches {
        let mut total_recall = 0.0f64;
        let search_start = Instant::now();

        for (q, gt) in queries.iter().zip(&ground_truths) {
            let hnsw_results: HashSet<String> = hnsw.search(q, k, Some(ef), None).unwrap()
                .into_iter()
                .map(|r| r.id.0)
                .collect();

            let intersection = hnsw_results.intersection(gt).count();
            total_recall += intersection as f64 / k as f64;
        }

        let total_duration = search_start.elapsed();
        let avg_recall = total_recall / n_queries as f64;
        let avg_latency_us = (total_duration.as_secs_f64() * 1_000_000.0) / n_queries as f64;

        println!(
            "| {} | {:.2}% | {:.1} µs | {:?}",
            ef,
            avg_recall * 100.0,
            avg_latency_us,
            total_duration
        );
    }
    println!("\n评估完成。");
}
