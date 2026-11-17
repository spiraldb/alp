//! Register dictionary lookup for eight-entry ALP-RD dictionaries.

use super::CombineBuffers;
use std::arch::x86_64::*;

/// All accesses are unaligned and bounded by the codes and both buffers.
///
/// The caller must make sure that the CPU supports AVX2, either at compile time or with
/// runtime detection.
#[target_feature(enable = "avx2")]
pub(super) fn combine_codes_f32(
    mut buffers: CombineBuffers<'_, u32>,
    codes: &[u16],
    dictionary: &[u32; 8],
) {
    // SAFETY: the dictionary holds 32 readable bytes, with no alignment requirement.
    let table = unsafe { _mm256_loadu_si256(dictionary.as_ptr().cast()) };

    let vector_len = buffers.len().min(codes.len()) / 8 * 8;
    let (right, output) = buffers.pointers();

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= codes.len(). Read eight codes.
        let packed = unsafe { _mm_loadu_si128(codes.as_ptr().add(offset).cast()) };

        // VPERMD uses only the low three index bits, matching the scalar code mask.
        let indices = _mm256_cvtepu16_epi32(packed);
        let left = _mm256_permutevar8x32_epi32(table, indices);

        // SAFETY: offset + 8 <= both buffer lengths. Read and write eight values.
        unsafe {
            let low = _mm256_loadu_si256(right.add(offset).cast());
            _mm256_storeu_si256(output.add(offset).cast(), _mm256_or_si256(left, low));
        }
    }

    buffers.combine_tail(codes, dictionary, vector_len);
}

/// Permute the two halves of each dictionary entry separately.
///
/// The caller must make sure that the CPU supports AVX2, either at compile time or with
/// runtime detection.
#[target_feature(enable = "avx2")]
pub(super) fn combine_codes(
    mut buffers: CombineBuffers<'_, u64>,
    codes: &[u16],
    dictionary: &[u64; 8],
) {
    let low = dictionary.map(|entry| entry as u32);
    let high = dictionary.map(|entry| (entry >> 32) as u32);

    // SAFETY: both arrays hold 32 readable bytes, with no alignment requirement.
    let (lo_table, hi_table) = unsafe {
        (
            _mm256_loadu_si256(low.as_ptr().cast()),
            _mm256_loadu_si256(high.as_ptr().cast()),
        )
    };

    let vector_len = buffers.len().min(codes.len()) / 8 * 8;
    let (right, output) = buffers.pointers();

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= codes.len(). Read eight codes.
        let packed = unsafe { _mm_loadu_si128(codes.as_ptr().add(offset).cast()) };

        let indices = _mm256_cvtepu16_epi32(packed);
        let lo = _mm256_permutevar8x32_epi32(lo_table, indices);
        let hi = _mm256_permutevar8x32_epi32(hi_table, indices);

        let even = _mm256_unpacklo_epi32(lo, hi);
        let odd = _mm256_unpackhi_epi32(lo, hi);
        let first = _mm256_permute2x128_si256::<0x20>(even, odd);
        let second = _mm256_permute2x128_si256::<0x31>(even, odd);

        // SAFETY: offset + 8 <= both buffer lengths. Read and write eight values.
        unsafe {
            let low = _mm256_loadu_si256(right.add(offset).cast());
            _mm256_storeu_si256(output.add(offset).cast(), _mm256_or_si256(first, low));

            let low = _mm256_loadu_si256(right.add(offset + 4).cast());
            _mm256_storeu_si256(output.add(offset + 4).cast(), _mm256_or_si256(second, low));
        }
    }

    buffers.combine_tail(codes, dictionary, vector_len);
}
