mod bitpack;

use crate::Exceptions;
use fastlanes::BitPacking;
use num_traits::{Float, One, PrimInt, Unsigned, Zero};
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::marker::PhantomData;
use std::ops::{Shl, Shr};

/// Returns the number of bits required to represent `value`, with a minimum of one.
#[inline]
pub const fn bit_width(value: u64) -> u8 {
    if value == 0 {
        1
    } else {
        value.ilog2().wrapping_add(1) as u8
    }
}

/// Maximum number of bits to cut from the MSB section of each float.
pub const CUT_LIMIT: usize = 16;

/// Maximum number of entries in the left-parts dictionary.
pub const MAX_DICT_SIZE: u8 = 8;

/// Maximum number of samples examined during dictionary search.
///
/// When the input is at least `2 * MAX_SAMPLE` elements long, [`find_best_dictionary`] strides
/// through it so that the search costs O(`MAX_SAMPLE`) rather than O(N). Below that threshold
/// the stride is one and every element is examined.
///
/// 4096 samples is enough to identify the dominant left-bit patterns: in practice the top-8
/// patterns emerge within the first few hundred values, and the chosen `right_bit_width` matches
/// the full-scan result (see `test_subsampling_matches_full_cut_point`).
const MAX_SAMPLE: usize = 4096;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Main trait for ALP-RD encodable floating-point numbers.
///
/// Like the paper, we limit this to the IEEE 754 single-precision (`f32`) and double-precision
/// (`f64`) floating-point types.
pub trait ALPRDFloat: private::Sealed + Float {
    /// The unsigned integer type with the same bit-width as the floating-point type.
    type UINT: PrimInt + BitPacking + Unsigned + One;

    /// Number of bits the value occupies in registers.
    const BITS: usize = size_of::<Self>() * 8;

    /// Transmutes bit-wise from the unsigned integer type to the floating-point type.
    fn from_bits(bits: Self::UINT) -> Self;

    /// Transmutes bit-wise into the unsigned integer type.
    fn to_bits(value: Self) -> Self::UINT;

    /// Converts the unsigned integer type to `u16`, truncating.
    fn to_u16(bits: Self::UINT) -> u16;

    /// Converts a `u16` to the unsigned integer type, widening.
    fn from_u16(value: u16) -> Self::UINT;
}

impl ALPRDFloat for f64 {
    type UINT = u64;

    #[inline]
    fn from_bits(bits: Self::UINT) -> Self {
        f64::from_bits(bits)
    }

    #[inline]
    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    #[inline]
    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    #[inline]
    fn from_u16(value: u16) -> Self::UINT {
        value as u64
    }
}

impl ALPRDFloat for f32 {
    type UINT = u32;

    #[inline]
    fn from_bits(bits: Self::UINT) -> Self {
        f32::from_bits(bits)
    }

    #[inline]
    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    #[inline]
    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    #[inline]
    fn from_u16(value: u16) -> Self::UINT {
        value as u32
    }
}

/// Encoder for ALP-RD ("real doubles") values.
///
/// The encoder calculates its parameters from a single sample of floating-point values,
/// and then can be applied to many vectors.
///
/// ALP-RD uses the algorithm outlined in Section 3.4 of the paper. The crux of it is that the
/// front (most significant) bits of many double vectors tend to be the same, i.e. most doubles in
/// a vector often use the same exponent and front bits. Compression proceeds by finding the best
/// prefix of up to 16 bits that can be collapsed into a dictionary of up to 8 elements. Each
/// double can then be broken into the front/left `L` bits, which neatly bit-packs down to 1-3 bits
/// per element (depending on the actual dictionary size). The remaining `R` bits naturally
/// bit-pack.
///
/// In the ideal case, this scheme allows us to store a sequence of doubles in 49 bits-per-value.
///
/// Our implementation draws on the MIT-licensed [C++ implementation] provided by the original
/// authors.
///
/// [C++ implementation]: https://github.com/cwida/ALP/blob/main/include/alp/rd.hpp
pub struct RDEncoder {
    /// Number of bits kept in the right (LSB) half of each float's bit representation.
    right_bit_width: u8,
    /// Forward mapping: `codes[code]` is the raw left-bit pattern that `code` encodes.
    codes: Vec<u16>,
    /// Reverse lookup: a raw left-bit pattern indexes the table, and the entry holds its
    /// dictionary code + 1, or zero if the pattern is not in the dictionary.
    ///
    /// Heap-allocated, 64KB. Built once per encoder, replacing the O(dictionary size) linear
    /// scan that [`RDEncoder::split_parts`] would otherwise run for every element.
    lookup: Box<[u8; 65536]>,
}

/// Builds the reverse lookup table used to dictionary-encode left parts.
///
/// The `code + 1` sentinel encoding lets a zero entry mean "not in the dictionary" without
/// widening the table to hold an `Option`, keeping it at 64KB.
///
/// # Panics
///
/// Panics if `codes` holds more than [`MAX_DICT_SIZE`] entries.
fn build_lookup(codes: &[u16]) -> Box<[u8; 65536]> {
    assert!(
        codes.len() <= MAX_DICT_SIZE as usize,
        "RDEncoder dictionary larger than MAX_DICT_SIZE"
    );

    // Heap-allocated, to avoid placing 64KB on the stack.
    let mut lookup = vec![0u8; 65536];
    for (code, &bits) in codes.iter().enumerate() {
        // `code + 1` fits in a u8 because `codes.len() <= MAX_DICT_SIZE`.
        let code = u8::try_from(code + 1).expect("code + 1 must fit in a u8");
        lookup[bits as usize] = code;
    }

    // Every dictionary entry must be reachable through the table it just populated. Written as a
    // round-trip through `codes` rather than an index comparison so that a dictionary holding a
    // repeated pattern, which `from_parts` accepts, does not trip the check.
    #[cfg(debug_assertions)]
    for &bits in codes {
        let code_plus_one = lookup[bits as usize];
        debug_assert_ne!(code_plus_one, 0, "dictionary pattern {bits} not in lookup");
        debug_assert_eq!(
            codes[code_plus_one as usize - 1],
            bits,
            "lookup must round-trip to the pattern it was built from"
        );
    }

    lookup
        .into_boxed_slice()
        .try_into()
        .expect("lookup table must be exactly 65536 bytes")
}

/// The "cut" ALP-RD vector.
///
/// ALP-RD splits a vector of input floating-point numbers into left parts and right parts,
/// divided at a cut point. The left and right values are held separately.
pub struct Split<F, U> {
    /// Dictionary codes for the left parts.
    left_parts: Vec<u16>,

    /// Exceptions for the `left_parts` that could not be dictionary encoded.
    left_exceptions: Exceptions<u16>,

    /// Dictionary for encoding the `left_parts`, held inline so that a split does not need a
    /// heap allocation for it. Only the first `left_dict_len` entries are meaningful.
    left_dict: [u16; MAX_DICT_SIZE as usize],

    /// Number of live entries in `left_dict`.
    left_dict_len: u8,

    /// Bit-width for the `left_parts` codes.
    left_parts_bit_width: u8,

    /// The right parts.
    right_parts: Vec<U>,

    /// Bit-width for the `right_parts` component.
    right_parts_bit_width: u8,

    phantom_data: PhantomData<F>,
}

impl<T, U> Split<T, U> {
    /// Consumes the split into its raw components.
    ///
    /// Returns `(left_parts, left_dict, left_exceptions, right_parts, right_bit_width)`:
    /// - `left_parts`: dictionary codes for the MSB halves, one per input value.
    /// - `left_dict`: the dictionary mapping a code to its raw left-bit pattern.
    /// - `left_exceptions`: values and positions that were not dictionary-encodable.
    /// - `right_parts`: LSB halves, one per input value, each `right_bit_width` bits wide.
    /// - `right_bit_width`: number of bits in each right-part element.
    pub fn into_parts(self) -> (Vec<u16>, Vec<u16>, Exceptions<u16>, Vec<U>, u8) {
        debug_assert!(
            self.left_dict_len <= MAX_DICT_SIZE,
            "left_dict_len exceeds MAX_DICT_SIZE"
        );

        // Materialise the inline dictionary into a `Vec` only here, on the rare path.
        let left_dict = self.left_dict[..self.left_dict_len as usize].to_vec();
        (
            self.left_parts,
            left_dict,
            self.left_exceptions,
            self.right_parts,
            self.right_parts_bit_width,
        )
    }

    /// Returns the dictionary codes of the left parts.
    pub fn left_parts(&self) -> &[u16] {
        &self.left_parts
    }

    /// Returns the dictionary used to encode the left parts.
    pub fn left_dict(&self) -> &[u16] {
        debug_assert!(
            self.left_dict_len <= MAX_DICT_SIZE,
            "left_dict_len exceeds MAX_DICT_SIZE"
        );
        &self.left_dict[..self.left_dict_len as usize]
    }

    /// Returns the exceptions of the left parts.
    pub fn left_exceptions(&self) -> &Exceptions<u16> {
        &self.left_exceptions
    }

    /// Returns the right parts.
    pub fn right_parts(&self) -> &[U] {
        &self.right_parts
    }

    /// Returns the bit-width of just the left parts, i.e. the width of the dictionary codes.
    pub fn left_parts_bit_width(&self) -> u8 {
        self.left_parts_bit_width
    }

    /// Returns the bit-width of just the right parts.
    pub fn right_parts_bit_width(&self) -> u8 {
        self.right_parts_bit_width
    }
}

impl<F, U> Split<F, U>
where
    F: ALPRDFloat<UINT = U>,
{
    /// Decodes back into a vector of the floating-point type.
    pub fn decode(&self) -> Vec<F> {
        alp_rd_decode(
            &self.left_parts,
            self.left_dict(),
            self.right_parts_bit_width,
            &self.right_parts,
            &self.left_exceptions.positions,
            &self.left_exceptions.values,
        )
    }
}

impl RDEncoder {
    /// Builds a new encoder from a sample of doubles.
    ///
    /// When `sample` is at least `2 * MAX_SAMPLE` elements long, the dictionary search strides
    /// through it so that it examines at most `MAX_SAMPLE` elements rather than every element.
    ///
    /// # Panics
    ///
    /// Panics if `sample` is empty.
    pub fn new<T>(sample: &[T]) -> Self
    where
        T: ALPRDFloat,
    {
        assert!(
            !sample.is_empty(),
            "ALP-RD requires a non-empty sample to build a dictionary"
        );

        let dictionary = find_best_dictionary::<T>(sample);

        let mut codes = vec![0; dictionary.dictionary.len()];
        dictionary.dictionary.into_iter().for_each(|(bits, code)| {
            // Write the reverse mapping into the codes vector.
            codes[code as usize] = bits
        });

        let lookup = build_lookup(&codes);

        Self {
            right_bit_width: dictionary.right_bit_width,
            codes,
            lookup,
        }
    }

    /// Builds a new encoder from known parameters.
    ///
    /// # Panics
    ///
    /// Panics if `codes` holds more than [`MAX_DICT_SIZE`] entries.
    pub fn from_parts(right_bit_width: u8, codes: Vec<u16>) -> Self {
        let lookup = build_lookup(&codes);

        Self {
            right_bit_width,
            codes,
            lookup,
        }
    }

    /// Returns the bit-width of the right (least significant) part of each value.
    #[inline]
    pub fn right_bit_width(&self) -> u8 {
        self.right_bit_width
    }

    /// Returns the bit-width of the dictionary codes of the left (most significant) parts.
    #[inline]
    pub fn left_bit_width(&self) -> u8 {
        bit_width(self.codes.len().saturating_sub(1) as u64)
    }

    /// Returns the dictionary of left parts, indexed by code.
    #[inline]
    pub fn codes(&self) -> &[u16] {
        &self.codes
    }

    /// Encodes the floating-point values into a [`Split`].
    pub fn split<T>(&self, doubles: &[T]) -> Split<T, T::UINT>
    where
        T: ALPRDFloat,
    {
        let (left_parts, right_parts, exception_pos, exception_values) = self.split_parts(doubles);

        // TODO(aduffy): pack the exception positions.
        let left_exceptions = Exceptions::new(exception_values, exception_pos);

        // `build_lookup` has already established that the dictionary fits inline.
        debug_assert!(
            self.codes.len() <= MAX_DICT_SIZE as usize,
            "dictionary must not exceed MAX_DICT_SIZE"
        );
        let mut left_dict = [0u16; MAX_DICT_SIZE as usize];
        left_dict[..self.codes.len()].copy_from_slice(&self.codes);

        Split {
            left_parts,
            left_dict,
            left_dict_len: self.codes.len() as u8,
            left_exceptions,
            left_parts_bit_width: self.left_bit_width(),
            right_parts,
            right_parts_bit_width: self.right_bit_width,
            phantom_data: PhantomData,
        }
    }

    /// Splits the floating-point values into their dictionary-encoded left parts, their right
    /// parts, and the positions and values of the left parts that are not in the dictionary.
    ///
    /// The left parts are returned as dictionary codes, packable into
    /// [`Self::left_bit_width`] bits; the right parts are packable into
    /// [`Self::right_bit_width`] bits. Positions of exceptions hold a code of zero.
    pub fn split_parts<T>(&self, doubles: &[T]) -> (Vec<u16>, Vec<T::UINT>, Vec<u64>, Vec<u16>)
    where
        T: ALPRDFloat,
    {
        assert!(
            !self.codes.is_empty(),
            "codes lookup table must be populated before RD encoding"
        );

        let mut left_parts: Vec<u16> = Vec::with_capacity(doubles.len());
        let mut right_parts: Vec<T::UINT> = Vec::with_capacity(doubles.len());
        let mut exception_pos: Vec<u64> = Vec::with_capacity(doubles.len() / 4);
        let mut exception_values: Vec<u16> = Vec::with_capacity(doubles.len() / 4);

        // Mask for the right parts.
        let right_mask = T::UINT::one().shl(self.right_bit_width as _) - T::UINT::one();

        // Single pass: split each value into its halves, dictionary-encode the left half through
        // the reverse lookup table, and record any pattern missing from the dictionary as an
        // exception.
        for (idx, v) in doubles.iter().copied().enumerate() {
            let bits = T::to_bits(v);
            right_parts.push(bits & right_mask);

            let left_raw = <T as ALPRDFloat>::to_u16(bits.shr(self.right_bit_width as _));
            let code_plus_one = self.lookup[left_raw as usize];
            if code_plus_one != 0 {
                left_parts.push(u16::from(code_plus_one) - 1);
            } else {
                exception_values.push(left_raw);
                exception_pos.push(idx as u64);

                left_parts.push(0u16);
            }
        }

        (left_parts, right_parts, exception_pos, exception_values)
    }
}

/// Decodes a vector of ALP-RD encoded values back into their original floating-point format.
///
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length, or if `exc_pos` and `exceptions`
/// differ in length.
pub fn alp_rd_decode<T: ALPRDFloat>(
    left_parts: &[u16],
    left_parts_dict: &[u16],
    right_bit_width: u8,
    right_parts: &[T::UINT],
    exc_pos: &[u64],
    exceptions: &[u16],
) -> Vec<T> {
    assert_eq!(
        left_parts.len(),
        right_parts.len(),
        "alp_rd_decode: left_parts.len != right_parts.len"
    );

    assert_eq!(
        exc_pos.len(),
        exceptions.len(),
        "alp_rd_decode: exc_pos.len != exceptions.len"
    );

    let mut decoded: Vec<T::UINT> = right_parts.to_vec();

    if exc_pos.is_empty() {
        // Non-patched fast path: every code maps through the dictionary, so we can
        // pre-shift the entire dictionary once and reduce the per-element hot loop to
        // a single table lookup + OR.
        alp_rd_combine_codes_inplace::<T>(
            &mut decoded,
            left_parts,
            left_parts_dict,
            right_bit_width,
        );
    } else {
        // Patched path: some left-part codes map to exception values that live outside
        // the dictionary. We must dictionary-decode first, then overwrite the exceptions,
        // before we can combine with right-parts.
        let mut left_parts = left_parts.to_vec();
        alp_rd_dict_decode_inplace(&mut left_parts, left_parts_dict);
        alp_rd_apply_patches(&mut left_parts, exc_pos, exceptions, 0);
        alp_rd_combine_inplace::<T>(&mut decoded, &left_parts, right_bit_width);
    }

    decoded.into_iter().map(T::from_bits).collect()
}

/// Replaces each dictionary code in `left_parts` with the left bit-pattern it encodes.
///
/// # Panics
///
/// Panics if `left_parts` contains a code that is not in `left_parts_dict`.
#[inline]
pub fn alp_rd_dict_decode_inplace(left_parts: &mut [u16], left_parts_dict: &[u16]) {
    for code in left_parts.iter_mut() {
        *code = left_parts_dict[*code as usize];
    }
}

/// Overwrites the exception positions of already dictionary-decoded `left_parts` with their true
/// left bit-patterns.
///
/// `offset` is subtracted from every index, to support patches that are stored relative to the
/// start of an unsliced array.
///
/// # Panics
///
/// Panics if `indices` and `patch_values` differ in length, or if an index is out of bounds.
#[inline]
pub fn alp_rd_apply_patches<I: PrimInt>(
    left_parts: &mut [u16],
    indices: &[I],
    patch_values: &[u16],
    offset: usize,
) {
    assert_eq!(
        indices.len(),
        patch_values.len(),
        "alp_rd_apply_patches: indices.len != patch_values.len"
    );

    indices
        .iter()
        .copied()
        .zip(patch_values.iter().copied())
        .for_each(|(idx, value)| {
            let idx = idx
                .to_usize()
                .expect("alp_rd_apply_patches: index out of range")
                - offset;
            left_parts[idx] = value;
        });
}

/// Combines dictionary-decoded `left_parts` into `right_parts` in-place, so that each element of
/// `right_parts` holds the bit-pattern of the original float.
///
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length.
#[inline]
pub fn alp_rd_combine_inplace<T: ALPRDFloat>(
    right_parts: &mut [T::UINT],
    left_parts: &[u16],
    right_bit_width: u8,
) {
    assert_eq!(
        left_parts.len(),
        right_parts.len(),
        "alp_rd_combine_inplace: left_parts.len != right_parts.len"
    );

    let shift = right_bit_width as usize;
    for (right, left) in right_parts.iter_mut().zip(left_parts.iter().copied()) {
        *right = (<T as ALPRDFloat>::from_u16(left) << shift) | *right;
    }
}

/// Combines dictionary-encoded left parts into `right_parts` in-place, so that each element of
/// `right_parts` holds the bit-pattern of the original float.
///
/// This is the unpatched fast path: the dictionary is pre-shifted once, reducing the hot loop to
/// a table lookup and an OR. Codes are masked into the dictionary, so codes beyond the dictionary
/// size decode to garbage rather than panicking.
///
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length, or if the dictionary holds more
/// than [`MAX_DICT_SIZE`] entries.
#[inline]
pub fn alp_rd_combine_codes_inplace<T: ALPRDFloat>(
    right_parts: &mut [T::UINT],
    left_parts: &[u16],
    left_parts_dict: &[u16],
    right_bit_width: u8,
) {
    assert_eq!(
        left_parts.len(),
        right_parts.len(),
        "alp_rd_combine_codes_inplace: left_parts.len != right_parts.len"
    );
    assert!(
        left_parts_dict.len() <= MAX_DICT_SIZE as usize,
        "alp_rd_combine_codes_inplace: dictionary larger than MAX_DICT_SIZE"
    );

    let shift = right_bit_width as usize;
    let mut shifted_dict = [T::UINT::zero(); MAX_DICT_SIZE as usize];
    for (i, &entry) in left_parts_dict.iter().enumerate() {
        shifted_dict[i] = <T as ALPRDFloat>::from_u16(entry) << shift;
    }

    // Masking keeps the lookup in-bounds without a branch; codes are < dictionary size by
    // construction.
    const CODE_MASK: usize = MAX_DICT_SIZE as usize - 1;
    for (right, code) in right_parts.iter_mut().zip(left_parts.iter().copied()) {
        *right = shifted_dict[(code as usize) & CODE_MASK] | *right;
    }
}

/// Finds the best "cut point" for a set of floating-point values, i.e. the one with the lowest
/// estimated compressed size.
///
/// All [`CUT_LIMIT`] candidate cut points are tried. Each counting pass is O(`MAX_SAMPLE`) once
/// `samples` is long enough to be strided, and the 256KB counting buffer is allocated once and
/// reused across every trial.
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T]) -> ALPRDDictionary {
    let stride = (samples.len() / MAX_SAMPLE).max(1);
    let effective_count = samples.len().div_ceil(stride);

    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();

    // Allocated once; `build_left_parts_dictionary` clears it on entry.
    let mut counts = vec![0u32; 65536];

    for p in 1..=CUT_LIMIT {
        let candidate_right_bw = (T::BITS - p) as u8;
        let (dictionary, exception_count) = build_left_parts_dictionary::<T>(
            samples,
            stride,
            candidate_right_bw,
            MAX_DICT_SIZE,
            &mut counts,
        );
        let estimated_size = estimate_compression_size(
            dictionary.right_bit_width,
            dictionary.left_bit_width,
            exception_count,
            effective_count,
        );
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = dictionary;
        }
    }

    best_dict
}

/// Builds a dictionary of the leftmost bits, counting pattern frequencies in a direct-addressed
/// array.
///
/// Left-bit patterns are `u16` values, so a pattern indexes the frequency array directly — no
/// hashing and no collisions, O(1) per element. `counts` is supplied by the caller so that the
/// 256KB buffer is allocated only once across all cut-point trials; it is cleared on entry.
///
/// Only every `stride`-th sample is counted.
fn build_left_parts_dictionary<T: ALPRDFloat>(
    samples: &[T],
    stride: usize,
    right_bw: u8,
    max_dict_size: u8,
    counts: &mut [u32],
) -> (ALPRDDictionary, usize) {
    assert!(
        right_bw >= (T::BITS - CUT_LIMIT) as _,
        "left-parts must be <= 16 bits"
    );

    counts.fill(0);

    // Count the number of occurrences of each left bit pattern.
    samples
        .iter()
        .step_by(stride)
        .copied()
        .map(|v| <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(right_bw as _)))
        .for_each(|item| counts[item as usize] += 1);

    // Collect the patterns that actually occurred, sorting so that heavy hitters come first.
    let mut sorted_bit_counts: Vec<(u16, u32)> = counts
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, count)| count > 0)
        .map(|(bits, count)| (bits as u16, count))
        .collect();
    sorted_bit_counts.sort_unstable_by_key(|(_, count)| Reverse(*count));

    // Assign the most-frequently occurring left-bits as dictionary codes, up to `dict_size`...
    let mut dictionary =
        FxHashMap::with_capacity_and_hasher(max_dict_size as _, Default::default());
    let mut code = 0u16;
    while code < (max_dict_size as _) && (code as usize) < sorted_bit_counts.len() {
        let (bits, _) = sorted_bit_counts[code as usize];
        dictionary.insert(bits, code);
        code += 1;
    }

    // ...and the rest are exceptions.
    let exception_count: usize = sorted_bit_counts
        .iter()
        .skip(code as _)
        .map(|(_, count)| *count as usize)
        .sum();

    // Left bit-width is determined based on the actual dictionary size.
    let max_code = dictionary.len().saturating_sub(1);
    let left_bw = bit_width(max_code as u64);

    (
        ALPRDDictionary {
            dictionary,
            right_bit_width: right_bw,
            left_bit_width: left_bw,
        },
        exception_count,
    )
}

/// Estimates the bits-per-value when using these compression settings.
fn estimate_compression_size(
    right_bw: u8,
    left_bw: u8,
    exception_count: usize,
    sample_n: usize,
) -> f64 {
    const EXC_POSITION_SIZE: usize = 16; // 16 bits to store the exception position.
    const EXC_SIZE: usize = 16; // up to 16 front bits per exception value.

    let exceptions_size = exception_count * (EXC_POSITION_SIZE + EXC_SIZE);
    (right_bw as f64) + (left_bw as f64) + ((exceptions_size as f64) / (sample_n as f64))
}

/// The ALP-RD dictionary, encoding the "left parts" and their dictionary encoding.
#[derive(Debug, Default)]
struct ALPRDDictionary {
    /// Items in the dictionary are bit patterns, along with their 16-bit encoding.
    dictionary: FxHashMap<u16, u16>,
    /// The (compressed) left bit-width. This is after bit-packing the dictionary codes.
    left_bit_width: u8,
    /// The right bit-width. This is the bit-packed width of each of the "real double" values.
    right_bit_width: u8,
}

#[cfg(test)]
mod test {
    use super::MAX_SAMPLE;
    use crate::{
        MAX_DICT_SIZE, RDEncoder, alp_rd_apply_patches, alp_rd_combine_codes_inplace,
        alp_rd_combine_inplace, alp_rd_decode, alp_rd_dict_decode_inplace, bit_width,
    };

    #[test]
    fn test_encode_decode() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];

        let encoder = RDEncoder::new(&values);

        let split = encoder.split(&values);
        let decoded = split.decode();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_encode_decode_with_exceptions() {
        // The outlier has a different exponent, so its left parts are not in the dictionary.
        let values = vec![0.1f64, 0.2f64, 3e100f64];
        let encoder = RDEncoder::new(&values[0..2]);

        let split = encoder.split(&values);
        assert_eq!(split.left_exceptions().positions(), &[2]);
        assert_eq!(split.decode(), values);
    }

    #[test]
    fn test_encode_decode_f32() {
        let values = vec![0.1f32, 0.2f32, 3e25f32];
        let encoder = RDEncoder::new(&values[0..2]);

        let split = encoder.split(&values);
        assert_eq!(split.left_exceptions().positions(), &[2]);
        assert_eq!(split.decode(), values);
    }

    #[test]
    fn test_from_parts_round_trips() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];
        let encoder = RDEncoder::new(&values);

        let rebuilt = RDEncoder::from_parts(encoder.right_bit_width(), encoder.codes().to_vec());
        assert_eq!(rebuilt.right_bit_width(), encoder.right_bit_width());
        assert_eq!(rebuilt.codes(), encoder.codes());
        assert_eq!(rebuilt.split(&values).decode(), values);
    }

    #[test]
    fn test_bit_widths() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];
        let encoder = RDEncoder::new(&values);
        let split = encoder.split(&values);

        assert!(encoder.codes().len() <= MAX_DICT_SIZE as usize);
        assert_eq!(split.left_parts_bit_width(), encoder.left_bit_width());
        assert_eq!(
            split.left_parts_bit_width() as usize,
            bit_width((encoder.codes().len() - 1) as u64) as usize
        );
        assert_eq!(split.right_parts_bit_width(), encoder.right_bit_width());
        assert!(
            split
                .right_parts()
                .iter()
                .all(|v| *v < (1u64 << split.right_parts_bit_width()))
        );
    }

    #[test]
    fn test_decode_primitives_match() {
        let values = vec![0.1f64, 0.2f64, 3e100f64];
        let encoder = RDEncoder::new(&values[0..2]);
        let (left_parts, right_parts, exc_pos, exc_values) = encoder.split_parts(&values);
        let dict = encoder.codes();
        let right_bit_width = encoder.right_bit_width();

        // Piecewise decode, as a consumer holding its own buffers would do it.
        let mut left = left_parts.clone();
        alp_rd_dict_decode_inplace(&mut left, dict);
        alp_rd_apply_patches(&mut left, &exc_pos, &exc_values, 0);
        let mut combined = right_parts.clone();
        alp_rd_combine_inplace::<f64>(&mut combined, &left, right_bit_width);
        let decoded: Vec<f64> = combined.into_iter().map(f64::from_bits).collect();

        assert_eq!(
            decoded,
            alp_rd_decode::<f64>(
                &left_parts,
                dict,
                right_bit_width,
                &right_parts,
                &exc_pos,
                &exc_values
            )
        );
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_combine_codes_matches_combine() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];
        let encoder = RDEncoder::new(&values);
        let (left_parts, right_parts, exc_pos, _) = encoder.split_parts(&values);
        assert!(exc_pos.is_empty());

        let mut fast = right_parts.clone();
        alp_rd_combine_codes_inplace::<f64>(
            &mut fast,
            &left_parts,
            encoder.codes(),
            encoder.right_bit_width(),
        );

        let mut left = left_parts;
        alp_rd_dict_decode_inplace(&mut left, encoder.codes());
        let mut slow = right_parts;
        alp_rd_combine_inplace::<f64>(&mut slow, &left, encoder.right_bit_width());

        assert_eq!(fast, slow);
        assert_eq!(
            fast.into_iter().map(f64::from_bits).collect::<Vec<_>>(),
            values
        );
    }

    #[test]
    fn test_apply_patches_with_offset() {
        // Patch indices are relative to the start of the unsliced array.
        let mut left = vec![0u16; 3];
        alp_rd_apply_patches(&mut left, &[10u64, 12], &[7u16, 9], 10);
        assert_eq!(left, vec![7, 0, 9]);
    }

    /// Values that miss the dictionary must still be recovered, via the exception path.
    #[test]
    fn test_exception_path_roundtrip() {
        // Train on values sharing one dominant pattern, then append a value whose bits differ
        // drastically so that it cannot be in the dictionary.
        let outlier = f64::from_bits(0xFFFF_0000_0000_0000);
        let mut training: Vec<f64> = vec![1.0f64; MAX_DICT_SIZE as usize + 1];
        training.push(outlier);

        let encoder = RDEncoder::new(&training);
        let split = encoder.split(&training);
        let decoded = split.decode();

        assert_eq!(decoded.len(), training.len());
        assert_eq!(
            f64::to_bits(decoded[decoded.len() - 1]),
            f64::to_bits(outlier),
            "exception-path value must decode to its original bits"
        );
    }

    /// Once the input exceeds `2 * MAX_SAMPLE` the dictionary search strides; the roundtrip must
    /// stay exact regardless.
    #[test]
    fn test_large_input_roundtrip() {
        // `2 * MAX_SAMPLE + 1` guarantees a stride greater than one.
        let n = 2 * MAX_SAMPLE + 1;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 1000.0).collect();

        let encoder = RDEncoder::new(&values);
        for chunk in values.chunks(1024) {
            assert_eq!(
                encoder.split(chunk).decode(),
                chunk,
                "chunk roundtrip must be exact"
            );
        }
    }

    /// `into_parts` materialises the inline dictionary into a `Vec`; the result must be usable to
    /// decode through the public entry point.
    #[test]
    fn test_into_parts_dict_materialisation() {
        let values = vec![1.5f64, 2.5f64, 3.5f64, 1.5f64];
        let encoder = RDEncoder::new(&values);
        let split = encoder.split(&values);

        let right_bit_width = split.right_parts_bit_width();
        assert_eq!(split.left_dict(), encoder.codes());

        let (left_parts, left_dict, left_exceptions, right_parts, bw) = split.into_parts();

        assert_eq!(bw, right_bit_width, "right_bit_width must be consistent");
        assert_eq!(left_dict, encoder.codes());
        assert_eq!(left_parts.len(), values.len());
        assert_eq!(right_parts.len(), values.len());

        let decoded = alp_rd_decode::<f64>(
            &left_parts,
            &left_dict,
            bw,
            &right_parts,
            left_exceptions.positions(),
            left_exceptions.values(),
        );
        assert_eq!(decoded, values);
    }

    /// The strided dictionary search must pick the same cut point as a full scan.
    ///
    /// When a float dataset has a stable distribution of left-bit patterns — the same few
    /// exponent and sign combinations recurring throughout — any `MAX_SAMPLE`-element subset
    /// identifies the dominant patterns as reliably as scanning everything. Demonstrated here on
    /// pseudo-random log-normal data, realistic for scientific datasets: the encoder built on an
    /// unstrided `MAX_SAMPLE` prefix and the one built on `3 * MAX_SAMPLE` values (which strides
    /// by three internally) agree on `right_bit_width`, and the strided encoder still roundtrips
    /// bit-exactly.
    #[test]
    fn test_subsampling_matches_full_cut_point() {
        // An inline LCG keeps this dependency-free. Any fixed-size prefix of its output is
        // statistically representative of the whole sequence.
        let mut seed: u64 = 0x517C_C1B7_2722_0A95;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let t = (seed >> 33) as f32 / u32::MAX as f32; // [0, 1)
            (t * 6.0 - 1.0).exp() * 1000.0 // log-normal, like much scientific float data
        };

        // n > 2 * MAX_SAMPLE, so find_best_dictionary strides by n / MAX_SAMPLE == 3.
        let n = 3 * MAX_SAMPLE + 1;
        let values: Vec<f32> = (0..n).map(|_| next_f32()).collect();

        // Built on exactly the first MAX_SAMPLE elements, so stride == 1.
        let encoder_prefix = RDEncoder::new(&values[..MAX_SAMPLE]);
        // Built on all n elements, striding internally over roughly as many elements.
        let encoder_strided = RDEncoder::new(&values);

        let chunk = &values[..64];
        assert_eq!(
            encoder_prefix.split(chunk).right_parts_bit_width(),
            encoder_strided.split(chunk).right_parts_bit_width(),
            "strided encoder must choose the same right_bit_width as the unstrided prefix encoder"
        );

        for (orig, dec) in chunk
            .iter()
            .zip(encoder_strided.split(chunk).decode().iter())
        {
            assert_eq!(
                f32::to_bits(*orig),
                f32::to_bits(*dec),
                "strided encoder must produce a bit-exact roundtrip"
            );
        }
    }

    #[test]
    fn test_bit_width_fn() {
        assert_eq!(bit_width(0), 1);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(u64::MAX), 64);
    }
}
