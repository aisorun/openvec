/// SIMD accelerated distance computations
///
/// x86_64: AVX2 / SSE4.1
/// aarch64: NEON

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use crate::types::VectorRef;
    use std::arch::x86_64::*;

    // ── AVX2 Implementation (processes 8 f32s at a time) ───────────────────────────────

    /// AVX2 L2 distance
    ///
    /// # Safety
    /// Must confirm CPU supports AVX2 before calling (guaranteed by select_impl in mod.rs)
    pub fn l2_avx2(a: VectorRef, b: VectorRef) -> f32 {
        // Safety: select_impl only selects this function when is_x86_feature_detected!("avx2") is true
        unsafe { l2_avx2_impl(a, b) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l2_avx2_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum
        let mut result = hsum_avx(sum);

        // Process tail elements (scalar)
        let offset = chunks * 8;
        for i in 0..remainder {
            let d = a[offset + i] - b[offset + i];
            result += d * d;
        }

        result.sqrt()
    }

    /// AVX2 Cosine distance
    pub fn cosine_avx2(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { cosine_avx2_impl(a, b) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cosine_avx2_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        let mut dot = hsum_avx(dot_sum);
        let mut norm_a = hsum_avx(norm_a_sum);
        let mut norm_b = hsum_avx(norm_b_sum);

        let offset = chunks * 8;
        for i in 0..remainder {
            let x = a[offset + i];
            let y = b[offset + i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            return 1.0;
        }
        1.0 - (dot / denom).clamp(-1.0, 1.0)
    }

    /// AVX2 Dot Product distance
    pub fn dot_avx2(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { dot_avx2_impl(a, b) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dot_avx2_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut result = hsum_avx(sum);

        let offset = chunks * 8;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }

        -result
    }

    /// AVX2 Horizontal sum (compress 8 f32s into 1)
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        // Add the high 128-bit to the low 128-bit
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        // Horizontal sum within 128-bit
        let shuf = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sum64);
        let sum32 = _mm_add_ss(sum64, shuf2);
        _mm_cvtss_f32(sum32)
    }

    // ── SSE4.1 Implementation (processes 4 f32s at a time) ───────────────────────────────

    /// SSE L2 distance
    pub fn l2_sse(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { l2_sse_impl(a, b) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn l2_sse_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        let mut result = hsum_sse(sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            let d = a[offset + i] - b[offset + i];
            result += d * d;
        }

        result.sqrt()
    }

    /// SSE Cosine distance
    pub fn cosine_sse(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { cosine_sse_impl(a, b) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn cosine_sse_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut dot_sum = _mm_setzero_ps();
        let mut norm_a_sum = _mm_setzero_ps();
        let mut norm_b_sum = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
            norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
            norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
        }

        let mut dot = hsum_sse(dot_sum);
        let mut norm_a = hsum_sse(norm_a_sum);
        let mut norm_b = hsum_sse(norm_b_sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            let x = a[offset + i];
            let y = b[offset + i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            return 1.0;
        }
        1.0 - (dot / denom).clamp(-1.0, 1.0)
    }

    /// SSE Dot Product distance
    pub fn dot_sse(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { dot_sse_impl(a, b) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn dot_sse_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        }

        let mut result = hsum_sse(sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }

        -result
    }

    /// SSE Horizontal sum (compress 4 f32s into 1)
    #[target_feature(enable = "sse4.1")]
    unsafe fn hsum_sse(v: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(v);
        let sum = _mm_add_ps(v, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sum);
        let sum2 = _mm_add_ss(sum, shuf2);
        _mm_cvtss_f32(sum2)
    }
}

// ── ARM NEON Implementation (aarch64, Apple Silicon / ARM64 servers) ─────────────

#[cfg(target_arch = "aarch64")]
pub mod arm {
    use crate::types::VectorRef;
    use std::arch::aarch64::*;

    /// NEON L2 distance
    pub fn l2_neon(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { l2_neon_impl(a, b) }
    }

    unsafe fn l2_neon_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        let mut result = vaddvq_f32(sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            let d = a[offset + i] - b[offset + i];
            result += d * d;
        }

        result.sqrt()
    }

    /// NEON Cosine distance
    pub fn cosine_neon(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { cosine_neon_impl(a, b) }
    }

    unsafe fn cosine_neon_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            dot_sum = vfmaq_f32(dot_sum, va, vb);
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        }

        let mut dot = vaddvq_f32(dot_sum);
        let mut norm_a = vaddvq_f32(norm_a_sum);
        let mut norm_b = vaddvq_f32(norm_b_sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            let x = a[offset + i];
            let y = b[offset + i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            return 1.0;
        }
        1.0 - (dot / denom).clamp(-1.0, 1.0)
    }

    /// NEON Dot Product distance
    pub fn dot_neon(a: VectorRef, b: VectorRef) -> f32 {
        unsafe { dot_neon_impl(a, b) }
    }

    unsafe fn dot_neon_impl(a: VectorRef, b: VectorRef) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut result = vaddvq_f32(sum);

        let offset = chunks * 4;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }

        -result
    }
}
