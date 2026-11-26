//! Register dictionary lookup for the eight-entry ALP-RD f64 dictionary.

use super::CombineBuffers;
use std::arch::x86_64::{
    __m128i, __m512i, _mm_loadu_si128, _mm512_cvtepu16_epi64, _mm512_loadu_si512, _mm512_or_si512,
    _mm512_permutexvar_epi64, _mm512_storeu_si512,
};

/// All loads and stores are unaligned and stay within the slices.
///
/// The caller must make sure that the CPU supports AVX-512F, either at compile time or with
/// runtime detection.
#[target_feature(enable = "avx512f")]
pub(super) fn combine_codes(
    mut buffers: CombineBuffers<'_, u64>,
    codes: &[u16],
    dictionary: &[u64; 8],
) {
    debug_assert_eq!(buffers.len(), codes.len());

    // SAFETY: the dictionary contains exactly 64 readable bytes; no alignment is required.
    let table = unsafe { _mm512_loadu_si512(dictionary.as_ptr().cast::<__m512i>()) };

    // The safe trait hook is callable directly, so keep mismatched lengths memory-safe
    // even without the public API's length check. `min` needs no panic branch.
    let vector_len = buffers.len().min(codes.len()) / 8 * 8;
    let (right, output) = buffers.pointers();

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= codes.len(). Load eight u16 codes (16 bytes).
        let packed = unsafe { _mm_loadu_si128(codes.as_ptr().add(offset).cast::<__m128i>()) };

        // Widen the codes to eight u64 indices. VPERMQ uses only the low three index bits,
        // matching the scalar code mask.
        let indices = _mm512_cvtepu16_epi64(packed);
        let left = _mm512_permutexvar_epi64(indices, table);

        // SAFETY: offset + 8 <= both buffer lengths. Read and write eight u64s (64 bytes).
        unsafe {
            let low = _mm512_loadu_si512(right.add(offset).cast::<__m512i>());
            _mm512_storeu_si512(
                output.add(offset).cast::<__m512i>(),
                _mm512_or_si512(left, low),
            );
        }
    }

    buffers.combine_tail(codes, dictionary, vector_len);
}
