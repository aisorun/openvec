use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use openvec_core::types::{DistanceMetric, Document, DocumentId, SearchRequest};
use openvec_core::index::{VectorIndex, flat::FlatIndex, hnsw::{HnswConfig, HnswIndex}};
use rand::prelude::*;

fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.random::<f32>()).collect()).collect()
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Search");

    for n in [1_000, 10_000, 50_000] {
        let dim = 128;
        let vecs = random_vecs(n, dim, 42);
        let query = random_vecs(1, dim, 99)[0].clone();

        let mut idx = HnswIndex::new(
            HnswConfig::new(dim, DistanceMetric::Cosine)
                .with_m(16)
                .with_ef_construction(100)
                .with_ef_search(50)
        );

        for (i, v) in vecs.iter().enumerate() {
            idx.insert(&DocumentId::from(format!("doc_{i}").as_str()), v).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("k=10", n), &query, |bench, q| {
            bench.iter(|| idx.search(q, 10, None).unwrap());
        });
    }

    group.finish();
}

fn bench_flat_vs_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat vs HNSW (1K vectors, dim=64)");
    let n = 1_000;
    let dim = 64;
    let vecs = random_vecs(n, dim, 42);
    let query = random_vecs(1, dim, 99)[0].clone();

    let mut flat = FlatIndex::new(dim, DistanceMetric::L2);
    let mut hnsw = HnswIndex::with_defaults(dim, DistanceMetric::L2);

    for (i, v) in vecs.iter().enumerate() {
        let id = DocumentId::from(format!("doc_{i}").as_str());
        flat.insert(&id, v).unwrap();
        hnsw.insert(&id, v).unwrap();
    }

    group.bench_function("flat/k=10", |bench| {
        bench.iter(|| flat.search(&query, 10, None).unwrap());
    });
    group.bench_function("hnsw/k=10", |bench| {
        bench.iter(|| hnsw.search(&query, 10, None).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_hnsw_search, bench_flat_vs_hnsw);
criterion_main!(benches);
