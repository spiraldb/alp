//! NEON register lookup for eight-entry ALP-RD dictionaries.

use std::arch::aarch64::*;

/// `dictionary` must contain u16 entries shifted left by `shift`, which must be below 64.
#[inline]
pub(super) unsafe fn combine_codes(
    right: &mut [u64],
    codes: &[u16],
    dictionary: &[u64; 8],
    shift: usize,
) {
    debug_assert_eq!(right.len(), codes.len());
    let packed = dictionary.map(|entry| (entry >> shift) as u16);
    // SAFETY: all table and index loads read exactly 16 bytes from local arrays.
    let table = unsafe { vreinterpretq_u8_u16(vld1q_u16(packed.as_ptr())) };
    let byte_offsets =
        unsafe { vld1q_u8([0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1].as_ptr()) };
    // SAFETY: this module is compiled only when NEON is enabled.
    let shifts = unsafe { vdupq_n_s64(shift as i64) };
    // Keep direct calls through the safe trait hook memory-safe for mismatched lengths.
    let vector_len = right.len().min(codes.len()) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= both slice lengths. Each code load reads eight u16;
        // each right-part load/store reads/writes two u64 within this eight-element block.
        unsafe {
            let code = vandq_u16(vld1q_u16(codes.as_ptr().add(offset)), vdupq_n_u16(7));
            // Each masked code c becomes the byte indices [2*c, 2*c+1].
            let indices = vaddq_u8(
                vreinterpretq_u8_u16(vmulq_n_u16(code, 0x0202)),
                byte_offsets,
            );
            let left = vreinterpretq_u16_u8(vqtbl1q_u8(table, indices));
            let lo = vmovl_u16(vget_low_u16(left));
            let hi = vmovl_high_u16(left);
            let widened = [
                vmovl_u32(vget_low_u32(lo)),
                vmovl_high_u32(lo),
                vmovl_u32(vget_low_u32(hi)),
                vmovl_high_u32(hi),
            ];
            for (pair, value) in widened.into_iter().enumerate() {
                let dst = right.as_mut_ptr().add(offset + pair * 2);
                vst1q_u64(dst, vorrq_u64(vld1q_u64(dst), vshlq_u64(value, shifts)));
            }
        }
    }
    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}

/// `dictionary` must contain u16 entries shifted left by `shift`, which must be below 32.
#[inline]
pub(super) unsafe fn combine_codes_f32(
    right: &mut [u32],
    codes: &[u16],
    dictionary: &[u32; 8],
    shift: usize,
) {
    debug_assert_eq!(right.len(), codes.len());
    let packed = dictionary.map(|entry| (entry >> shift) as u16);
    // SAFETY: both local arrays contain exactly 16 readable bytes.
    let table = unsafe { vreinterpretq_u8_u16(vld1q_u16(packed.as_ptr())) };
    let byte_offsets =
        unsafe { vld1q_u8([0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1].as_ptr()) };
    // SAFETY: this module is compiled only when NEON is enabled.
    let shifts = unsafe { vdupq_n_s32(shift as i32) };
    // Keep direct calls through the safe trait hook memory-safe for mismatched lengths.
    let vector_len = right.len().min(codes.len()) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: this block reads eight codes and reads/writes eight right parts;
        // offset + 8 <= both slice lengths. No alignment is required.
        unsafe {
            let code = vandq_u16(vld1q_u16(codes.as_ptr().add(offset)), vdupq_n_u16(7));
            let indices = vaddq_u8(
                vreinterpretq_u8_u16(vmulq_n_u16(code, 0x0202)),
                byte_offsets,
            );
            let left = vreinterpretq_u16_u8(vqtbl1q_u8(table, indices));
            let lo = vshlq_u32(vmovl_u16(vget_low_u16(left)), shifts);
            let hi = vshlq_u32(vmovl_high_u16(left), shifts);
            let dst = right.as_mut_ptr().add(offset);
            vst1q_u32(dst, vorrq_u32(vld1q_u32(dst), lo));
            vst1q_u32(dst.add(4), vorrq_u32(vld1q_u32(dst.add(4)), hi));
        }
    }
    for (low, &code) in right[vector_len..].iter_mut().zip(&codes[vector_len..]) {
        *low |= dictionary[code as usize & 7];
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn neon_f32_lookup_matches_scalar() {
        for shift in 0..32 {
            for dict_len in 0..=8 {
                let mut dict = [0u32; 8];
                for (i, entry) in dict[..dict_len].iter_mut().enumerate() {
                    *entry = (0x8001u32 + i as u32 * 311) << shift;
                }
                for len in 0..=137 {
                    for misalignment in 0..8 {
                        let codes: Vec<u16> = (0..len + misalignment)
                            .map(|i| (i as u16).wrapping_mul(7919))
                            .collect();
                        let mut right: Vec<u32> = (0..len + misalignment)
                            .map(|i| (i as u32).wrapping_mul(0x9abc_def1))
                            .collect();
                        let mut expected = right.clone();
                        for (low, &code) in expected[misalignment..]
                            .iter_mut()
                            .zip(&codes[misalignment..])
                        {
                            *low |= dict[code as usize & 7];
                        }
                        // SAFETY: NEON is baseline on aarch64; dictionary, shift and
                        // slice lengths satisfy the kernel contract.
                        unsafe {
                            super::combine_codes_f32(
                                &mut right[misalignment..],
                                &codes[misalignment..],
                                &dict,
                                shift,
                            );
                        }
                        assert_eq!(right, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn neon_lookup_matches_scalar() {
        for shift in 0..64 {
            for dict_len in 0..=8 {
                let mut dict = [0u64; 8];
                for (i, entry) in dict[..dict_len].iter_mut().enumerate() {
                    *entry = (0x8001u64 + i as u64 * 311) << shift;
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
                        // SAFETY: NEON is baseline on aarch64, dictionary and shift satisfy
                        // the kernel contract, and the slices have identical lengths.
                        unsafe {
                            super::combine_codes(
                                &mut right[misalignment..],
                                &codes[misalignment..],
                                &dict,
                                shift,
                            );
                        }
                        assert_eq!(right, expected);
                    }
                }
            }
        }
    }
}
