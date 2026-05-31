/// Distance computation module
///
/// Supports three distance metrics:
/// - L2 (Euclidean distance)
/// - Cosine (Cosine distance, stored as 1 - similarity)
/// - DotProduct (Dot product, stored as -dot_product)
///
/// All distances are unified to the "smaller is better" semantic.
/// Automatically utilizes SIMD acceleration (AVX2 / SSE2 / NEON) on supported platforms.

mod scalar;
mod simd;

pub use scalar::*;

use crate::types::{DistanceMetric, VectorRef};



/// Distance Calculator
///
/// Selects the optimal computation strategy (detecting CPU features at runtime)
pub struct DistanceCalculator {
    metric: DistanceMetric,
    func: fn(VectorRef, VectorRef) -> f32,
    cosine_normalized: bool,
}

impl DistanceCalculator {
    /// Creates a calculator based on the metric, automatically selecting the optimal implementation
    pub fn new(metric: DistanceMetric) -> Self {
        let is_cosine = metric == DistanceMetric::Cosine;
        let actual_metric = if is_cosine { DistanceMetric::DotProduct } else { metric };
        let func = select_impl(actual_metric);
        Self {
            metric,
            func,
            cosine_normalized: is_cosine,
        }
    }

    /// Computes the distance between two vectors (smaller is better)
    #[inline(always)]
    pub fn compute(&self, a: VectorRef, b: VectorRef) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimension mismatch");
        let raw = (self.func)(a, b);
        if self.metric == DistanceMetric::Cosine && self.cosine_normalized {
            // raw is -dot(a,b). We want 1.0 - dot(a,b) = 1.0 + raw
            1.0 + raw
        } else {
            raw
        }
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Selects the optimal implementation based on CPU features and metric
fn select_impl(metric: DistanceMetric) -> fn(VectorRef, VectorRef) -> f32 {
    // x86_64: Runtime detection of AVX2 / SSE4.1
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return match metric {
            DistanceMetric::L2 => simd::x86::l2_avx2,
            DistanceMetric::Cosine => simd::x86::cosine_avx2,
            DistanceMetric::DotProduct => simd::x86::dot_avx2,
        };
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") {
        return match metric {
            DistanceMetric::L2 => simd::x86::l2_sse,
            DistanceMetric::Cosine => simd::x86::cosine_sse,
            DistanceMetric::DotProduct => simd::x86::dot_sse,
        };
    }

    // aarch64: NEON is always available (known at compile time)
    #[cfg(target_arch = "aarch64")]
    return match metric {
        DistanceMetric::L2 => simd::arm::l2_neon,
        DistanceMetric::Cosine => simd::arm::cosine_neon,
        DistanceMetric::DotProduct => simd::arm::dot_neon,
    };

    // Other platforms fallback to scalar implementation
    #[allow(unreachable_code)]
    match metric {
        DistanceMetric::L2 => scalar::l2_distance,
        DistanceMetric::Cosine => scalar::cosine_distance,
        DistanceMetric::DotProduct => scalar::dot_distance,
    }
}

/// Compute distance directly using scalar functions (for testing and validation)
pub mod raw {
    use crate::types::VectorRef;
    pub use super::scalar::{l2_distance, cosine_distance, dot_distance};

    /// Computes the squared L2 distance (avoids sqrt, useful for sorting)
    pub fn l2_distance_sq(a: VectorRef, b: VectorRef) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn vec(data: &[f32]) -> Vec<f32> {
        data.to_vec()
    }

    #[test]
    fn test_l2_identical() {
        let a = vec(&[1.0, 2.0, 3.0]);
        let calc = DistanceCalculator::new(DistanceMetric::L2);
        let dist = calc.compute(&a, &a);
        assert!(dist.abs() < EPS, "L2(a,a) should be 0, got {dist}");
    }

    #[test]
    fn test_l2_known_value() {
        let a = vec(&[0.0, 0.0]);
        let b = vec(&[3.0, 4.0]);
        let calc = DistanceCalculator::new(DistanceMetric::L2);
        let dist = calc.compute(&a, &b);
        assert!((dist - 5.0).abs() < EPS, "L2([0,0],[3,4]) should be 5, got {dist}");
    }

    #[test]
    fn test_cosine_identical_normalized() {
        // For normalized vectors, identical vectors have a cosine distance of 0
        let a = vec(&[1.0 / f32::sqrt(2.0), 1.0 / f32::sqrt(2.0)]);
        let calc = DistanceCalculator::new(DistanceMetric::Cosine);
        let dist = calc.compute(&a, &a);
        assert!(dist.abs() < EPS, "Cosine(a,a) should be 0, got {dist}");
    }

    #[test]
    fn test_cosine_orthogonal() {
        // Orthogonal vectors have a cosine distance of 1
        let a = vec(&[1.0, 0.0]);
        let b = vec(&[0.0, 1.0]);
        let calc = DistanceCalculator::new(DistanceMetric::Cosine);
        let dist = calc.compute(&a, &b);
        assert!((dist - 1.0).abs() < EPS, "Cosine([1,0],[0,1]) should be 1, got {dist}");
    }

    #[test]
    fn test_cosine_opposite() {
        // Opposite vectors have a cosine distance of 2
        let a = vec(&[1.0, 0.0]);
        let b = vec(&[-1.0, 0.0]);
        let calc = DistanceCalculator::new(DistanceMetric::Cosine);
        let dist = calc.compute(&a, &b);
        assert!((dist - 2.0).abs() < EPS, "Cosine(a,-a) should be 2, got {dist}");
    }

    #[test]
    fn test_dot_product() {
        let a = vec(&[1.0, 2.0, 3.0]);
        let b = vec(&[4.0, 5.0, 6.0]);
        let calc = DistanceCalculator::new(DistanceMetric::DotProduct);
        // dot = 1*4 + 2*5 + 3*6 = 32; stored as -32
        let dist = calc.compute(&a, &b);
        assert!((dist - (-32.0)).abs() < EPS, "DotProduct should be -32, got {dist}");
    }

    #[test]
    fn test_ordering_l2() {
        // Vector [2,0] is closer to [0,0] than [10,0]
        let query = vec(&[0.0, 0.0]);
        let near = vec(&[2.0, 0.0]);
        let far = vec(&[10.0, 0.0]);
        let calc = DistanceCalculator::new(DistanceMetric::L2);
        assert!(calc.compute(&query, &near) < calc.compute(&query, &far));
    }

    #[test]
    fn test_ordering_cosine() {
        let query = vec(&[1.0, 0.0]);
        let near = vec(&[1.0, 0.1]);   // Close
        let far = vec(&[0.1, 1.0]);    // Diverged
        let calc = DistanceCalculator::new(DistanceMetric::Cosine);
        assert!(calc.compute(&query, &near) < calc.compute(&query, &far));
    }
}
