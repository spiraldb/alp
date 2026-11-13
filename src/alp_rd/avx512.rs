//! Register dictionary lookup for the eight-entry ALP-RD f64 dictionary.

use std::arch::x86_64::{
    __m128i, __m512i, _mm_loadu_si128, _mm512_cvtepu16_epi64, _mm512_loadu_si512, _mm512_or_si512,
    _mm512_permutexvar_epi64, _mm512_storeu_si512,
};

/// Requires AVX-512F. All loads and stores are unaligned and stay within the slices.
#[inline]
pub(super) unsafe fn combine_codes(right: &mut [u64], codes: &[u16], dictionary: &[u64; 8]) {
    debug_assert_eq!(right.len(), codes.len());
    // SAFETY: the dictionary contains exactly 64 readable bytes; no alignment is required.
    let table = unsafe { _mm512_loadu_si512(dictionary.as_ptr().cast::<__m512i>()) };
    // The safe trait hook is callable directly, so keep mismatched lengths memory-safe
    // even without the public API's length check. `min` needs no panic branch.
    let vector_len = right.len().min(codes.len()) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= both slice lengths. Load eight u16 codes (16 bytes),
        // widen them to eight u64 indices, and load/store eight u64 right parts (64 bytes).
        // VPERMQ uses only the low three index bits, matching the scalar code mask.
        unsafe {
            let packed = _mm_loadu_si128(codes.as_ptr().add(offset).cast::<__m128i>());
            let indices = _mm512_cvtepu16_epi64(packed);
            let left = _mm512_permutexvar_epi64(indices, table);
            let dst = right.as_mut_ptr().add(offset).cast::<__m512i>();
            let low = _mm512_loadu_si512(dst);
            _mm512_storeu_si512(dst, _mm512_or_si512(left, low));
        }
    }
    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vector_lookup_matches_scalar() {
        // Include zero-padded dictionaries and arbitrary high code bits, not just valid codes.
        for dict_len in 0..=8 {
            let mut dict = [0u64; 8];
            for (i, entry) in dict[..dict_len].iter_mut().enumerate() {
                *entry = (0x8001u64 + i as u64 * 311) << 48;
            }
            for len in 0..=137 {
                for misalignment in 0..8 {
                    let codes: Vec<u16> = (0..len + misalignment)
                        .map(|i| (i as u16).wrapping_mul(7919))
                        .collect();
                    let mut right: Vec<u64> = (0..len + misalignment)
                        .map(|i| (i as u64).wrapping_mul(0x1234_5678_9abc_def1))
                        .collect();
                    let mut expected = right.clone();
                    for (low, &code) in expected[misalignment..]
                        .iter_mut()
                        .zip(&codes[misalignment..])
                    {
                        *low |= dict[code as usize & 7];
                    }
                    // SAFETY: this module is compiled only when AVX-512F is enabled.
                    unsafe {
                        super::combine_codes(
                            &mut right[misalignment..],
                            &codes[misalignment..],
                            &dict,
                        );
                    }
                    assert_eq!(right, expected);
                }
            }
        }
    }
}
