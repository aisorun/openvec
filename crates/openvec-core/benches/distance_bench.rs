use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use openvec_core::distance::{DistanceCalculator, raw};
use openvec_core::types::DistanceMetric;

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 Distance");

    for dim in [64, 128, 256, 512, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 / dim as f32).collect();

        group.bench_with_input(BenchmarkId::new("scalar", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| raw::l2_distance(a, b));
        });

        let calc = DistanceCalculator::new(DistanceMetric::L2);
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| calc.compute(a, b));
        });
    }

    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Distance");

    for dim in [64, 128, 256, 512, 768] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let calc = DistanceCalculator::new(DistanceMetric::Cosine);
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| calc.compute(a, b));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_l2, bench_cosine);
criterion_main!(benches);
