//! Register dictionary lookup for eight-entry ALP-RD dictionaries.

use std::arch::x86_64::*;

/// Requires AVX2. All accesses are unaligned and bounded by both slices.
#[inline]
pub(super) unsafe fn combine_codes_f32(right: &mut [u32], codes: &[u16], dictionary: &[u32; 8]) {
    // SAFETY: the dictionary holds 32 readable bytes, with no alignment requirement.
    let table = unsafe { _mm256_loadu_si256(dictionary.as_ptr().cast()) };
    let vector_len = right.len().min(codes.len()) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: each iteration reads eight codes and reads/writes eight right parts.
        // VPERMD uses only the low three index bits, matching the scalar code mask.
        unsafe {
            let packed = _mm_loadu_si128(codes.as_ptr().add(offset).cast());
            let indices = _mm256_cvtepu16_epi32(packed);
            let left = _mm256_permutevar8x32_epi32(table, indices);
            let dst = right.as_mut_ptr().add(offset).cast();
            _mm256_storeu_si256(dst, _mm256_or_si256(left, _mm256_loadu_si256(dst)));
        }
    }
    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}

#[cfg(not(target_feature = "avx512f"))]
/// Requires AVX2. Permute the two halves of each dictionary entry separately.
#[inline]
pub(super) unsafe fn combine_codes(right: &mut [u64], codes: &[u16], dictionary: &[u64; 8]) {
    let low = dictionary.map(|entry| entry as u32);
    let high = dictionary.map(|entry| (entry >> 32) as u32);
    // SAFETY: both arrays hold 32 readable bytes, with no alignment requirement.
    let (lo_table, hi_table) = unsafe {
        (
            _mm256_loadu_si256(low.as_ptr().cast()),
            _mm256_loadu_si256(high.as_ptr().cast()),
        )
    };
    let vector_len = right.len().min(codes.len()) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: each iteration reads eight codes and reads/writes eight right parts.
        unsafe {
            let packed = _mm_loadu_si128(codes.as_ptr().add(offset).cast());
            let indices = _mm256_cvtepu16_epi32(packed);
            let lo = _mm256_permutevar8x32_epi32(lo_table, indices);
            let hi = _mm256_permutevar8x32_epi32(hi_table, indices);
            let even = _mm256_unpacklo_epi32(lo, hi);
            let odd = _mm256_unpackhi_epi32(lo, hi);
            let first = _mm256_permute2x128_si256::<0x20>(even, odd);
            let second = _mm256_permute2x128_si256::<0x31>(even, odd);
            let dst = right.as_mut_ptr().add(offset).cast();
            _mm256_storeu_si256(dst, _mm256_or_si256(first, _mm256_loadu_si256(dst)));
            let dst = right.as_mut_ptr().add(offset + 4).cast();
            _mm256_storeu_si256(dst, _mm256_or_si256(second, _mm256_loadu_si256(dst)));
        }
    }
    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}
