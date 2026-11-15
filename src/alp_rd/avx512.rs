//! Register dictionary lookup for the eight-entry ALP-RD f64 dictionary.

use std::arch::x86_64::{
    __m128i, __m512i, _mm_loadu_si128, _mm512_cvtepu16_epi64, _mm512_loadu_si512, _mm512_or_si512,
    _mm512_permutexvar_epi64, _mm512_storeu_si512,
};

/// All loads and stores are unaligned and stay within the slices.
///
/// The caller must make sure that the CPU supports AVX-512F, either at compile time or with
/// runtime detection.
#[target_feature(enable = "avx512f")]
pub(super) fn combine_codes(right: &mut [u64], codes: &[u16], dictionary: &[u64; 8]) {
    debug_assert_eq!(right.len(), codes.len());

    // SAFETY: the dictionary contains exactly 64 readable bytes; no alignment is required.
    let table = unsafe { _mm512_loadu_si512(dictionary.as_ptr().cast::<__m512i>()) };

    // The safe trait hook is callable directly, so keep mismatched lengths memory-safe
    // even without the public API's length check. `min` needs no panic branch.
    let vector_len = right.len().min(codes.len()) / 8 * 8;

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= both slice lengths. Load eight u16 codes (16 bytes).
        let packed = unsafe { _mm_loadu_si128(codes.as_ptr().add(offset).cast::<__m128i>()) };

        // Widen the codes to eight u64 indices. VPERMQ uses only the low three index bits,
        // matching the scalar code mask.
        let indices = _mm512_cvtepu16_epi64(packed);
        let left = _mm512_permutexvar_epi64(indices, table);

        // SAFETY: offset + 8 <= right.len(). Load and store eight u64 right parts (64 bytes).
        unsafe {
            let dst = right.as_mut_ptr().add(offset).cast::<__m512i>();
            _mm512_storeu_si512(dst, _mm512_or_si512(left, _mm512_loadu_si512(dst)));
        }
    }

    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}
