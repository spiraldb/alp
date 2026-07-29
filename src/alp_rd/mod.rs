mod bitpack;

use crate::Exceptions;
use fastlanes::BitPacking;
use num_traits::{Float, One, PrimInt, Unsigned, Zero};
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::marker::PhantomData;
use std::ops::{Range, Shl, Shr};

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

/// Number of distinct left-part bit patterns, i.e. `2^CUT_LIMIT`.
///
/// Every candidate cut point keeps at most [`CUT_LIMIT`] bits, so a left part always fits in a
/// `u16` and can be used directly as an index into a table of this size.
const LEFT_PARTS_CARDINALITY: usize = 1 << CUT_LIMIT;

/// Target number of values examined when searching for the best cut point.
///
/// Dictionary search evaluates all [`CUT_LIMIT`] candidate cut points, so its cost is proportional
/// to `CUT_LIMIT * sample_size`. Capping the sample makes [`RDEncoder::new`] cost roughly the same
/// for a million values as for a few thousand: the dominant left-bit patterns of a column follow
/// from its exponent distribution, which a few thousand values already characterise.
///
/// Patterns the search never sees are not lost — they simply become exceptions in
/// [`RDEncoder::split`], at a cost the size estimate already accounts for.
const MAX_SAMPLE: usize = 4096;

/// Length of each contiguous run of values taken by [`SamplePlan::subsample`].
///
/// Runs, rather than a fixed stride, are what make subsampling safe. A stride of `len / MAX_SAMPLE`
/// aliases with any periodicity in the input — interleaved coordinate or embedding columns,
/// round-robin sensor readings — and a strided sample then observes only one phase of the data.
/// Measured on interleaved values of two very different magnitudes, striding chose cut points 2.9
/// to 10.7 bits/value worse than a full scan, with 15-40% of the input left un-encodable. Runs come
/// within 0.03 bits/value of a full scan on the same input, and touch far fewer cache lines than a
/// wide stride. See `test_subsample_matches_full_scan_on_periodic_data`.
const SAMPLE_BLOCK: usize = 64;

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
    /// Reverse of `codes`: indexed by a left-part bit pattern, holding that pattern's code plus
    /// one, or zero when the pattern is not in the dictionary.
    ///
    /// The bias by one lets a single byte carry both the code and its presence, keeping the table
    /// at 64 KiB. Built once per encoder so that [`Self::split_parts`] needs one indexed load per
    /// value instead of a scan over `codes`.
    lookup: Box<[u8; LEFT_PARTS_CARDINALITY]>,
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

    /// Dictionary for encoding the `left_parts`, holding `left_dict_len` live entries.
    ///
    /// Stored inline because it never exceeds [`MAX_DICT_SIZE`] entries, so a split does not
    /// allocate for it. Reach for it through [`Split::left_dict`], which trims it to length.
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
    /// Consumes the parts of the result.
    pub fn into_parts(self) -> (Vec<u16>, Vec<u16>, Exceptions<u16>, Vec<U>, u8) {
        // The inline dictionary is materialised only here, for callers that want to own it.
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
    /// Long samples are not read in full: past [`MAX_SAMPLE`] values the search examines evenly
    /// spread contiguous runs instead (see [`SamplePlan`]), so this call costs about the same for a
    /// million values as for a few thousand. Values in the unexamined gaps still encode correctly —
    /// a left-part pattern the search never saw becomes an exception in [`Self::split`].
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

        let plan = SamplePlan::subsample(sample.len(), MAX_SAMPLE, SAMPLE_BLOCK);
        let dictionary = find_best_dictionary::<T>(sample, &plan);

        let mut codes = vec![0; dictionary.dictionary.len()];
        dictionary.dictionary.into_iter().for_each(|(bits, code)| {
            // Write the reverse mapping into the codes vector.
            codes[code as usize] = bits
        });

        Self::from_parts(dictionary.right_bit_width, codes)
    }

    /// Builds a new encoder from known parameters.
    ///
    /// # Panics
    ///
    /// Panics if `codes` holds more than [`MAX_DICT_SIZE`] entries.
    pub fn from_parts(right_bit_width: u8, codes: Vec<u16>) -> Self {
        // Beyond MAX_DICT_SIZE the codes no longer fit the bit-width the decoder assumes, and
        // `alp_rd_combine_codes_inplace` would mask them into the wrong dictionary slot.
        assert!(
            codes.len() <= MAX_DICT_SIZE as usize,
            "ALP-RD dictionary must hold at most MAX_DICT_SIZE entries"
        );

        let mut lookup: Box<[u8; LEFT_PARTS_CARDINALITY]> = vec![0u8; LEFT_PARTS_CARDINALITY]
            .into_boxed_slice()
            .try_into()
            .expect("lookup is allocated with exactly LEFT_PARTS_CARDINALITY entries");
        for (code, &bits) in codes.iter().enumerate() {
            // Keep the lowest code for a repeated pattern, matching a first-match-wins scan.
            let slot = &mut lookup[bits as usize];
            if *slot == 0 {
                *slot = code as u8 + 1;
            }
        }

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

        // `from_parts` bounds the dictionary at MAX_DICT_SIZE, so this always fits.
        let mut left_dict = [0u16; MAX_DICT_SIZE as usize];
        left_dict[..self.codes.len()].copy_from_slice(&self.codes);

        Split {
            left_parts,
            left_exceptions,
            left_dict,
            left_dict_len: self.codes.len() as u8,
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

        // Cut each value in two and dict-encode its left part in the same pass, keeping track of
        // exceptions. The dictionary lookup is one indexed load into `self.lookup`.
        for (idx, v) in doubles.iter().copied().enumerate() {
            let bits = T::to_bits(v);
            right_parts.push(bits & right_mask);

            let pattern = <T as ALPRDFloat>::to_u16(bits.shr(self.right_bit_width as _));
            let biased_code = self.lookup[pattern as usize];
            if biased_code != 0 {
                left_parts.push(u16::from(biased_code - 1));
            } else {
                // Exceptions hold a code of zero and carry their true pattern out-of-band.
                exception_values.push(pattern);
                exception_pos.push(idx as u64);
                left_parts.push(0);
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

/// Which elements of the input to examine when searching for the best cut point.
///
/// A plan is a set of disjoint, ascending, contiguous ranges. It is built once and shared by every
/// candidate cut point, so the trials all score the same values and their estimates stay
/// comparable.
#[derive(Debug)]
struct SamplePlan {
    ranges: Vec<Range<usize>>,
    count: usize,
}

impl SamplePlan {
    /// A plan covering every element of a `len`-element input.
    fn full(len: usize) -> Self {
        let mut ranges = Vec::new();
        if len > 0 {
            ranges.push(0..len);
        }
        Self { ranges, count: len }
    }

    /// A plan covering at most `max_sample` elements of a `len`-element input, as evenly spread
    /// contiguous runs of `block` values.
    ///
    /// Falls back to [`Self::full`] for inputs already at or below `max_sample`.
    fn subsample(len: usize, max_sample: usize, block: usize) -> Self {
        let block = block.clamp(1, max_sample.max(1));
        if len <= max_sample {
            return Self::full(len);
        }

        let n_blocks = (max_sample / block).max(1);
        // `len > max_sample >= n_blocks * block` puts the spacing at `block` or more, so the runs
        // stay disjoint, and the last one starts at `len - block` or earlier, so all are in bounds.
        // Multiplying the floored spacing (rather than dividing a product) keeps this from
        // overflowing on absurd lengths.
        let spacing = if n_blocks > 1 {
            (len - block) / (n_blocks - 1)
        } else {
            0
        };
        let ranges: Vec<Range<usize>> = (0..n_blocks)
            .map(|i| {
                let start = i * spacing;
                start..start + block
            })
            .collect();

        let count = ranges.iter().map(Range::len).sum();
        Self { ranges, count }
    }

    /// The ranges to sample, ascending and disjoint.
    fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Total number of elements the plan visits.
    fn count(&self) -> usize {
        self.count
    }
}

/// Buffers reused by [`build_left_parts_dictionary`] across every candidate cut point.
///
/// `counts` is direct-addressed by the 16-bit left part, so counting is one indexed increment per
/// value with no hashing and no collisions. The catch is that a 64K-entry table is far larger than
/// a typical sample, so clearing it or scanning it for non-zero entries would swamp the counting
/// itself — sixteen times over, once per trial. `touched` avoids that by recording which entries
/// were written, which bounds both the reset and the collection by the number of distinct patterns
/// actually seen.
struct DictSearchScratch {
    /// Occurrences of each left-part pattern, indexed by the pattern itself.
    ///
    /// Zero everywhere except at the patterns listed in `touched`.
    counts: Box<[u32; LEFT_PARTS_CARDINALITY]>,
    /// The patterns holding a non-zero entry in `counts`, in first-seen order.
    touched: Vec<u16>,
    /// `(pattern, count)` pairs for the current trial, heaviest hitters first.
    sorted: Vec<(u16, u32)>,
}

impl DictSearchScratch {
    fn new() -> Self {
        // Allocated as a Vec so the 256 KiB table is never a stack temporary.
        let counts: Box<[u32; LEFT_PARTS_CARDINALITY]> = vec![0u32; LEFT_PARTS_CARDINALITY]
            .into_boxed_slice()
            .try_into()
            .expect("counts is allocated with exactly LEFT_PARTS_CARDINALITY entries");

        Self {
            counts,
            touched: Vec::new(),
            sorted: Vec::new(),
        }
    }
}

/// Finds the best "cut point" for a set of floating-point values, i.e. the one with the lowest
/// estimated compressed size, considering only the elements `plan` selects.
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T], plan: &SamplePlan) -> ALPRDDictionary {
    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();
    let mut scratch = DictSearchScratch::new();

    for p in 1..=CUT_LIMIT {
        let candidate_right_bw = (T::BITS - p) as u8;
        let (dictionary, exception_count) = build_left_parts_dictionary::<T>(
            samples,
            plan,
            candidate_right_bw,
            MAX_DICT_SIZE,
            &mut scratch,
        );
        let estimated_size = estimate_compression_size(
            dictionary.right_bit_width,
            dictionary.left_bit_width,
            exception_count,
            // Per sampled element, not per input element, so that the exception term stays on the
            // same scale as the counts it came from.
            plan.count(),
        );
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = dictionary;
        }
    }

    best_dict
}

/// Builds a dictionary of the leftmost bits.
///
/// `scratch` must arrive with an all-zero `counts` and an empty `touched`, and is left that way.
fn build_left_parts_dictionary<T: ALPRDFloat>(
    samples: &[T],
    plan: &SamplePlan,
    right_bw: u8,
    max_dict_size: u8,
    scratch: &mut DictSearchScratch,
) -> (ALPRDDictionary, usize) {
    assert!(
        right_bw >= (T::BITS - CUT_LIMIT) as _,
        "left-parts must be <= 16 bits"
    );

    let DictSearchScratch {
        counts,
        touched,
        sorted,
    } = scratch;
    debug_assert!(
        touched.is_empty(),
        "counts was left dirty by the previous trial"
    );

    // Count the number of occurrences of each left bit pattern. The pattern is a u16, so it indexes
    // `counts` directly.
    for range in plan.ranges() {
        for value in &samples[range.start..range.end] {
            let pattern = <T as ALPRDFloat>::to_u16(T::to_bits(*value).shr(right_bw as _));
            let count = &mut counts[pattern as usize];
            if *count == 0 {
                touched.push(pattern);
            }
            *count += 1;
        }
    }

    // Sorted counts: heavy hitters first, ties broken on the pattern so that the dictionary — and
    // so the encoded output — is reproducible rather than left to the sort's internals.
    sorted.clear();
    sorted.extend(
        touched
            .iter()
            .map(|&pattern| (pattern, counts[pattern as usize])),
    );
    sorted.sort_unstable_by_key(|&(pattern, count)| (Reverse(count), pattern));

    // Assign the most-frequently occurring left-bits as dictionary codes, up to `max_dict_size`...
    let dict_len = (max_dict_size as usize).min(sorted.len());
    let mut dictionary = FxHashMap::with_capacity_and_hasher(dict_len, Default::default());
    for (code, &(bits, _)) in sorted[..dict_len].iter().enumerate() {
        dictionary.insert(bits, code as u16);
    }

    // ...and the rest are exceptions.
    let exception_count: usize = sorted[dict_len..]
        .iter()
        .map(|&(_, count)| count as usize)
        .sum();

    // Left bit-width is determined based on the actual dictionary size.
    let left_bw = bit_width(dictionary.len().saturating_sub(1) as u64);

    // Restore the all-zero invariant, touching only the entries this trial wrote.
    for pattern in touched.drain(..) {
        counts[pattern as usize] = 0;
    }

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
    use super::{MAX_SAMPLE, SAMPLE_BLOCK, SamplePlan, find_best_dictionary};
    use crate::{
        MAX_DICT_SIZE, RDEncoder, alp_rd_apply_patches, alp_rd_combine_codes_inplace,
        alp_rd_combine_inplace, alp_rd_decode, alp_rd_dict_decode_inplace, bit_width,
    };

    /// A small linear congruential generator, so the tests need no extra dependencies.
    struct Lcg(u64);

    impl Lcg {
        fn new() -> Self {
            Self(0x517C_C1B7_2722_0A95)
        }

        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// A log-normal-ish draw, spanning enough exponents to look like real scientific data.
        fn next_log_normal(&mut self) -> f64 {
            (self.next_unit() * 6.0 - 1.0).exp() * 1000.0
        }
    }

    /// What a cut point actually costs over a whole input.
    struct CutPointCost {
        /// Compressed bits per value, exceptions included at the same weight the encoder's own
        /// estimate uses.
        bits_per_value: f64,
        /// Fraction of values whose left parts fall outside the best dictionary at this cut point.
        exception_rate: f64,
    }

    /// Measures what choosing `right_bw` costs over all of `values`.
    ///
    /// Sampling strategies are compared on this rather than on the `right_bit_width` they name.
    /// The width is only a proxy: adjacent cut points routinely score within a rounding error of
    /// each other, so two samplers can name different widths and still compress identically well.
    /// What matters is the cost of the choice.
    fn cut_point_cost(values: &[f64], right_bw: u8) -> CutPointCost {
        // Position plus value, matching `estimate_compression_size`.
        const EXCEPTION_BITS: f64 = 32.0;

        let mut counts = std::collections::HashMap::new();
        for value in values {
            *counts
                .entry((value.to_bits() >> right_bw) as u16)
                .or_insert(0usize) += 1;
        }
        let mut sorted: Vec<usize> = counts.into_values().collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));

        let dict_len = (MAX_DICT_SIZE as usize).min(sorted.len());
        let encodable: usize = sorted.iter().take(dict_len).sum();
        let exception_rate = 1.0 - (encodable as f64 / values.len() as f64);
        let left_bw = bit_width(dict_len.saturating_sub(1) as u64);

        CutPointCost {
            bits_per_value: right_bw as f64 + left_bw as f64 + exception_rate * EXCEPTION_BITS,
            exception_rate,
        }
    }

    /// Compares what the subsampling encoder costs against what a full scan would have cost.
    fn subsample_vs_full_scan(values: &[f64]) -> (CutPointCost, CutPointCost) {
        let subsampled = RDEncoder::new(values).right_bit_width();
        let full =
            find_best_dictionary::<f64>(values, &SamplePlan::full(values.len())).right_bit_width;
        (
            cut_point_cost(values, subsampled),
            cut_point_cost(values, full),
        )
    }

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

    #[test]
    fn test_bit_width_fn() {
        assert_eq!(bit_width(0), 1);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(u64::MAX), 64);
    }

    #[test]
    #[should_panic(expected = "at most MAX_DICT_SIZE entries")]
    fn test_from_parts_rejects_oversized_dictionary() {
        // A larger dictionary needs codes wider than the decoder's fast path masks for.
        RDEncoder::from_parts(52, vec![0u16; MAX_DICT_SIZE as usize + 1]);
    }

    #[test]
    fn test_sample_plan_covers_short_inputs_fully() {
        for len in [0usize, 1, 63, 64, 4095, MAX_SAMPLE] {
            let plan = SamplePlan::subsample(len, MAX_SAMPLE, SAMPLE_BLOCK);
            assert_eq!(plan.count(), len, "short inputs must be scanned in full");
            let covered: usize = plan.ranges().iter().map(|r| r.len()).sum();
            assert_eq!(covered, len);
        }
    }

    #[test]
    fn test_sample_plan_ranges_are_in_bounds_and_disjoint() {
        // Includes lengths that are and are not multiples of the block and sample sizes.
        for len in [
            MAX_SAMPLE + 1,
            2 * MAX_SAMPLE,
            3 * MAX_SAMPLE + 1,
            100_003,
            1 << 20,
        ] {
            let plan = SamplePlan::subsample(len, MAX_SAMPLE, SAMPLE_BLOCK);
            assert!(
                plan.count() <= MAX_SAMPLE,
                "subsampling must honour the MAX_SAMPLE budget for len {len}"
            );
            assert_eq!(
                plan.count(),
                plan.ranges().iter().map(|r| r.len()).sum::<usize>()
            );

            let mut prev_end = 0;
            for range in plan.ranges() {
                assert!(
                    range.start >= prev_end,
                    "ranges must be ascending, disjoint"
                );
                assert!(
                    range.end <= len,
                    "range {}..{} exceeds {len}",
                    range.start,
                    range.end
                );
                prev_end = range.end;
            }
            assert!(!plan.ranges().is_empty());
        }
    }

    /// Subsampling must not cost compression on data whose distribution is uniform throughout,
    /// which is the easy case.
    #[test]
    fn test_subsample_matches_full_scan_on_random_data() {
        let mut rng = Lcg::new();
        let values: Vec<f64> = (0..8 * MAX_SAMPLE).map(|_| rng.next_log_normal()).collect();

        let (subsampled, full) = subsample_vs_full_scan(&values);
        assert!(
            subsampled.bits_per_value <= full.bits_per_value + 0.1,
            "subsampling cost {:.3} bits/value against a full scan's {:.3}",
            subsampled.bits_per_value,
            full.bits_per_value
        );
    }

    /// Regression test for sampling that aliases with periodic input.
    ///
    /// Interleaving values of two very different magnitudes — flattened coordinate or embedding
    /// columns, round-robin sensor readings — gives the input a period. A sampler striding by
    /// `len / MAX_SAMPLE` lands on one phase whenever that stride is a multiple of the period, so
    /// the search sees only one magnitude and picks a cut point that is badly wrong for the whole
    /// input: measured at 2.9 to 10.7 bits/value worse than a full scan, with exception rates of
    /// 15% to 40%. Contiguous runs see every phase and land within 0.03 bits/value of a full scan.
    #[test]
    fn test_subsample_matches_full_scan_on_periodic_data() {
        for period in [2usize, 3, 4, 8] {
            // Size the input so that a striding sampler would use a stride of exactly `period`.
            let len = period * MAX_SAMPLE;
            let mut rng = Lcg::new();
            let values: Vec<f64> = (0..len)
                .map(|i| {
                    let magnitude = if i % period == 0 { 1e-8 } else { 1e12 };
                    magnitude * (1.0 + rng.next_unit())
                })
                .collect();

            let (subsampled, full) = subsample_vs_full_scan(&values);
            assert!(
                subsampled.bits_per_value <= full.bits_per_value + 0.5,
                "period {period}: subsampling cost {:.3} bits/value against a full scan's {:.3}",
                subsampled.bits_per_value,
                full.bits_per_value
            );
            assert!(
                subsampled.exception_rate < 0.10,
                "period {period}: subsampling chose a cut point leaving {:.2}% of the input \
                 un-encodable",
                subsampled.exception_rate * 100.0
            );
        }
    }

    /// Every value must round-trip, including the ones the dictionary search never looked at.
    #[test]
    fn test_large_input_round_trips_in_chunks() {
        let mut rng = Lcg::new();
        let values: Vec<f64> = (0..4 * MAX_SAMPLE + 7)
            .map(|_| rng.next_log_normal())
            .collect();
        let encoder = RDEncoder::new(&values);

        for chunk in values.chunks(1024) {
            assert_eq!(&encoder.split(chunk).decode(), chunk);
        }
    }

    /// The dictionary must not depend on hash iteration order or on how the sort breaks ties.
    #[test]
    fn test_dictionary_is_reproducible() {
        // Every pattern appears the same number of times, so selection is decided entirely by the
        // tie-break.
        let values: Vec<f64> = (0..1024)
            .map(|i| f64::from_bits(((i % 32) as u64) << 52 | 1))
            .collect();

        let expected = RDEncoder::new(&values);
        for _ in 0..16 {
            let actual = RDEncoder::new(&values);
            assert_eq!(actual.codes(), expected.codes());
            assert_eq!(actual.right_bit_width(), expected.right_bit_width());
        }
    }

    /// The reverse lookup table used by `split_parts` must agree with the forward `codes` table.
    #[test]
    fn test_lookup_agrees_with_codes() {
        let mut rng = Lcg::new();
        let values: Vec<f64> = (0..2048).map(|_| rng.next_log_normal()).collect();
        let encoder = RDEncoder::new(&values);

        // Encoding the dictionary patterns themselves must yield their own codes, exception-free.
        let right_bw = encoder.right_bit_width();
        let patterns: Vec<f64> = encoder
            .codes()
            .iter()
            .map(|&bits| f64::from_bits((bits as u64) << right_bw))
            .collect();
        let (left_parts, _, exc_pos, _) = encoder.split_parts(&patterns);

        assert!(exc_pos.is_empty(), "dictionary patterns must not except");
        assert_eq!(
            left_parts,
            (0..encoder.codes().len() as u16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_dict_matches_into_parts() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];
        let encoder = RDEncoder::new(&values);
        let split = encoder.split(&values);

        let inline_dict = split.left_dict().to_vec();
        let (_, owned_dict, _, _, _) = split.into_parts();

        assert_eq!(inline_dict, owned_dict);
        assert_eq!(owned_dict, encoder.codes());
    }
}
