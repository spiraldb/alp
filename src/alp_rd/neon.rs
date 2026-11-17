//! NEON register lookup for eight-entry ALP-RD dictionaries.

use super::CombineBuffers;
use std::arch::aarch64::*;

/// `dictionary` must contain u16 entries shifted left by `shift`, which must be below 64.
#[inline]
pub(super) unsafe fn combine_codes(
    buffers: CombineBuffers<'_, u64>,
    codes: &[u16],
    dictionary: &[u64; 8],
    shift: usize,
) {
    // The encoder cuts at most 16 leading bits. Use widening shifts with immediate
    // operands for this common case; retain arbitrary shifts for the public API.
    // SAFETY: both specializations keep accesses within the input slices and
    // preserve the shifted-u16 dictionary contract.
    if shift >= 48 {
        unsafe { combine_codes_impl::<true>(buffers, codes, dictionary, shift) };
    } else {
        unsafe { combine_codes_impl::<false>(buffers, codes, dictionary, shift) };
    }
}

#[inline]
unsafe fn combine_codes_impl<const HIGH: bool>(
    mut buffers: CombineBuffers<'_, u64>,
    codes: &[u16],
    dictionary: &[u64; 8],
    shift: usize,
) {
    let packed = dictionary.map(|entry| (entry >> if HIGH { 48 } else { shift }) as u16);
    // SAFETY: all table and index loads read exactly 16 bytes from local arrays.
    let table = unsafe { vreinterpretq_u8_u16(vld1q_u16(packed.as_ptr())) };
    let byte_offsets =
        unsafe { vld1q_u8([0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1].as_ptr()) };
    // SAFETY: this module is compiled only when NEON is enabled.
    let shifts = unsafe { vdupq_n_s64(shift as i64) };
    // Keep direct calls through the safe trait hook memory-safe for mismatched lengths.
    let vector_len = buffers.len().min(codes.len()) / 8 * 8;
    let (right, output) = buffers.pointers();
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: offset + 8 <= the codes and both buffers. Each code load reads eight u16;
        // each right-part load/store reads/writes two u64 within this eight-element block.
        unsafe {
            let code = vandq_u16(vld1q_u16(codes.as_ptr().add(offset)), vdupq_n_u16(7));
            // Each masked code c becomes the byte indices [2*c, 2*c+1].
            let indices = vaddq_u8(
                vreinterpretq_u8_u16(vmulq_n_u16(code, 0x0202)),
                byte_offsets,
            );
            let left = vreinterpretq_u16_u8(vqtbl1q_u8(table, indices));
            let widened = if HIGH {
                let lo = vshll_n_u16::<16>(vget_low_u16(left));
                let hi = vshll_high_n_u16::<16>(left);
                [
                    vshll_n_u32::<32>(vget_low_u32(lo)),
                    vshll_high_n_u32::<32>(lo),
                    vshll_n_u32::<32>(vget_low_u32(hi)),
                    vshll_high_n_u32::<32>(hi),
                ]
            } else {
                let lo = vmovl_u16(vget_low_u16(left));
                let hi = vmovl_high_u16(left);
                [
                    vshlq_u64(vmovl_u32(vget_low_u32(lo)), shifts),
                    vshlq_u64(vmovl_high_u32(lo), shifts),
                    vshlq_u64(vmovl_u32(vget_low_u32(hi)), shifts),
                    vshlq_u64(vmovl_high_u32(hi), shifts),
                ]
            };
            for (pair, value) in widened.into_iter().enumerate() {
                let index = offset + pair * 2;
                let low = vld1q_u64(right.add(index));
                vst1q_u64(output.add(index), vorrq_u64(low, value));
            }
        }
    }
    buffers.combine_tail(codes, dictionary, vector_len);
}

/// `dictionary` must contain u16 entries shifted left by `shift`, which must be below 32.
#[inline]
pub(super) unsafe fn combine_codes_f32(
    buffers: CombineBuffers<'_, u32>,
    codes: &[u16],
    dictionary: &[u32; 8],
    shift: usize,
) {
    // SAFETY: both specializations keep accesses within the input slices.
    if shift >= 16 {
        unsafe { combine_codes_f32_impl::<true>(buffers, codes, dictionary, shift) };
    } else {
        unsafe { combine_codes_f32_impl::<false>(buffers, codes, dictionary, shift) };
    }
}

#[inline]
unsafe fn combine_codes_f32_impl<const HIGH: bool>(
    mut buffers: CombineBuffers<'_, u32>,
    codes: &[u16],
    dictionary: &[u32; 8],
    shift: usize,
) {
    let packed = dictionary.map(|entry| (entry >> if HIGH { 16 } else { shift }) as u16);
    // SAFETY: both local arrays contain exactly 16 readable bytes.
    let table = unsafe { vreinterpretq_u8_u16(vld1q_u16(packed.as_ptr())) };
    let byte_offsets =
        unsafe { vld1q_u8([0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1].as_ptr()) };
    // SAFETY: this module is compiled only when NEON is enabled.
    let shifts = unsafe { vdupq_n_s32(shift as i32) };
    // Keep direct calls through the safe trait hook memory-safe for mismatched lengths.
    let vector_len = buffers.len().min(codes.len()) / 8 * 8;
    let (right, output) = buffers.pointers();
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: this block reads eight codes and reads/writes eight right parts;
        // offset + 8 <= the codes and both buffers. No alignment is required.
        unsafe {
            let code = vandq_u16(vld1q_u16(codes.as_ptr().add(offset)), vdupq_n_u16(7));
            let indices = vaddq_u8(
                vreinterpretq_u8_u16(vmulq_n_u16(code, 0x0202)),
                byte_offsets,
            );
            let left = vreinterpretq_u16_u8(vqtbl1q_u8(table, indices));
            let (lo, hi) = if HIGH {
                (
                    vshll_n_u16::<16>(vget_low_u16(left)),
                    vshll_high_n_u16::<16>(left),
                )
            } else {
                (
                    vshlq_u32(vmovl_u16(vget_low_u16(left)), shifts),
                    vshlq_u32(vmovl_high_u16(left), shifts),
                )
            };
            vst1q_u32(
                output.add(offset),
                vorrq_u32(vld1q_u32(right.add(offset)), lo),
            );
            vst1q_u32(
                output.add(offset + 4),
                vorrq_u32(vld1q_u32(right.add(offset + 4)), hi),
            );
        }
    }
    buffers.combine_tail(codes, dictionary, vector_len);
}
