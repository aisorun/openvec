/// Scalar (non-SIMD) distance computation implementation
///
/// Serves as a fallback implementation on platforms that do not support SIMD.
/// Also used to process tail elements (remaining elements less than a SIMD register) in SIMD implementations.

use crate::types::VectorRef;

/// L2 Euclidean distance
///
/// `sqrt(Σ(aᵢ - bᵢ)²)`
#[inline]
pub fn l2_distance(a: VectorRef, b: VectorRef) -> f32 {
    let sq_sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| {
        let d = x - y;
        d * d
    }).sum();
    sq_sum.sqrt()
}

/// Cosine distance (smaller is better)
///
/// `1 - (a · b) / (|a| * |b|)`
///
/// Returns values in range [0, 2]:
/// - 0: Identical directions
/// - 1: Orthogonal
/// - 2: Completely opposite
#[inline]
pub fn cosine_distance(a: VectorRef, b: VectorRef) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < f32::EPSILON {
        return 1.0; // Zero vector is treated as orthogonal
    }

    // Clamp to valid range due to floating point precision
    let similarity = (dot / denom).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Dot product distance (smaller is better)
///
/// Stored as `-(a · b)`, so that higher similarity yields smaller distance.
#[inline]
pub fn dot_distance(a: VectorRef, b: VectorRef) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot
}

/// Squared L2 distance (avoids sqrt, only used for sorting comparisons)
#[inline]
pub fn l2_distance_sq(a: VectorRef, b: VectorRef) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = x - y;
        d * d
    }).sum()
}

/// Vector L2 norm
#[inline]
pub fn norm(v: VectorRef) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalizes vector (in-place)
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_zero() {
        let a = [1.0f32, 2.0, 3.0];
        assert!(l2_distance(&a, &a).abs() < EPS);
    }

    #[test]
    fn l2_345_triangle() {
        let a = [0.0f32, 0.0];
        let b = [3.0f32, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < EPS);
    }

    #[test]
    fn cosine_identical() {
        let a = [1.0f32, 1.0];
        assert!(cosine_distance(&a, &a).abs() < EPS);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < EPS);
    }

    #[test]
    fn cosine_opposite() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < EPS);
    }

    #[test]
    fn dot_product_known() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 32 → stored as -32
        assert!((dot_distance(&a, &b) - (-32.0)).abs() < EPS);
    }

    #[test]
    fn normalize_unit_vector() {
        let mut v = [3.0f32, 4.0];
        normalize(&mut v);
        assert!((norm(&v) - 1.0).abs() < EPS);
    }
}
