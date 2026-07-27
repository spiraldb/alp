mod bitpack;

use crate::Exceptions;
use fastlanes::BitPacking;
use num_traits::{Float, One, PrimInt, Unsigned, Zero};
use rustc_hash::FxHashMap;
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
    right_bit_width: u8,
    codes: Vec<u16>,
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

    /// Dictionary for encoding the `left_parts`.
    left_dict: Vec<u16>,

    /// Bit-width for the `left_parts` codes.
    left_parts_bit_width: u8,

    /// The right parts.
    right_parts: Vec<U>,

    /// Bit-width for the `right_parts` component.
    right_parts_bit_width: u8,

    phantom_data: PhantomData<F>,
}

impl<T, U> Split<T, U> {
    /// Consumes the parts of the result.
    pub fn into_parts(self) -> (Vec<u16>, Vec<u16>, Exceptions<u16>, Vec<U>, u8) {
        (
            self.left_parts,
            self.left_dict,
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
        &self.left_dict
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
            &self.left_dict,
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

        Self {
            right_bit_width: dictionary.right_bit_width,
            codes,
        }
    }

    /// Builds a new encoder from known parameters.
    #[inline]
    pub fn from_parts(right_bit_width: u8, codes: Vec<u16>) -> Self {
        Self {
            right_bit_width,
            codes,
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

        Split {
            left_parts,
            left_dict: self.codes.clone(),
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

        for v in doubles.iter().copied() {
            right_parts.push(T::to_bits(v) & right_mask);
            left_parts.push(<T as ALPRDFloat>::to_u16(
                T::to_bits(v).shr(self.right_bit_width as _),
            ));
        }

        // Dict-encode the left parts, keeping track of exceptions.
        for (idx, left) in left_parts.iter_mut().enumerate() {
            // TODO: revisit if we need to change the branch order for perf.
            if let Some(code) = self.codes.iter().position(|v| *v == *left) {
                *left = code as u16;
            } else {
                exception_values.push(*left);
                exception_pos.push(idx as _);

                *left = 0u16;
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
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T]) -> ALPRDDictionary {
    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();

    for p in 1..=CUT_LIMIT {
        let candidate_right_bw = (T::BITS - p) as u8;
        let (dictionary, exception_count) =
            build_left_parts_dictionary::<T>(samples, candidate_right_bw, MAX_DICT_SIZE);
        let estimated_size = estimate_compression_size(
            dictionary.right_bit_width,
            dictionary.left_bit_width,
            exception_count,
            samples.len(),
        );
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = dictionary;
        }
    }

    best_dict
}

/// Builds a dictionary of the leftmost bits.
fn build_left_parts_dictionary<T: ALPRDFloat>(
    samples: &[T],
    right_bw: u8,
    max_dict_size: u8,
) -> (ALPRDDictionary, usize) {
    assert!(
        right_bw >= (T::BITS - CUT_LIMIT) as _,
        "left-parts must be <= 16 bits"
    );

    // Count the number of occurrences of each left bit pattern.
    let mut counts = FxHashMap::default();
    samples
        .iter()
        .copied()
        .map(|v| <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(right_bw as _)))
        .for_each(|item| *counts.entry(item).or_default() += 1);

    // Sorted counts: sort by negative count so that heavy hitters sort first.
    let mut sorted_bit_counts: Vec<(u16, usize)> = counts.into_iter().collect();
    sorted_bit_counts.sort_by_key(|(_, count)| count.wrapping_neg());

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
        .map(|(_, count)| *count)
        .sum();

    // Left bit-width is determined based on the actual dictionary size.
    let max_code = dictionary.len() - 1;
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
    const EXC_POSITION_SIZE: usize = 16; // two bytes for exception position.
    const EXC_SIZE: usize = 16; // two bytes for each exception (up to 16 front bits).

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
    use crate::{
        alp_rd_apply_patches, alp_rd_combine_codes_inplace, alp_rd_combine_inplace, alp_rd_decode,
        alp_rd_dict_decode_inplace, bit_width, RDEncoder, MAX_DICT_SIZE,
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
        assert!(split
            .right_parts()
            .iter()
            .all(|v| *v < (1u64 << split.right_parts_bit_width())));
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

        let mut left = left_parts.clone();
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

    #[test]
    fn test_bit_width_fn() {
        assert_eq!(bit_width(0), 1);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(u64::MAX), 64);
    }
}
