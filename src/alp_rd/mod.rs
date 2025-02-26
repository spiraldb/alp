mod bitpack;

use crate::Exceptions;
use fastlanes::BitPacking;
use num_traits::{Float, One, PrimInt, Unsigned};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Shl, Shr};

macro_rules! bit_width {
    ($value:expr) => {
        if $value == 0 {
            1
        } else {
            $value.ilog2().wrapping_add(1) as usize
        }
    };
}

/// Max number of bits to cut from the MSB section of each float.
const CUT_LIMIT: usize = 16;

const MAX_DICT_SIZE: u8 = 8;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Main trait for ALP-RD encodable floating point numbers.
///
/// Like the paper, we limit this to the IEEE7 754 single-precision (`f32`) and double-precision
/// (`f64`) floating point types.
pub trait ALPRDFloat: private::Sealed + Float {
    /// The unsigned integer type with the same bit-width as the floating-point type.
    type UINT: PrimInt + BitPacking + Unsigned + One;

    /// Number of bits the value occupies in registers.
    const BITS: usize = size_of::<Self>() * 8;

    /// Bit-wise transmute from the unsigned integer type to the floating-point type.
    fn from_bits(bits: Self::UINT) -> Self;

    /// Bit-wise transmute into the unsigned integer type.
    fn to_bits(value: Self) -> Self::UINT;

    /// Truncating conversion from the unsigned integer type to `u16`.
    fn to_u16(bits: Self::UINT) -> u16;

    /// Type-widening conversion from `u16` to the unsigned integer type.
    fn from_u16(value: u16) -> Self::UINT;
}

impl ALPRDFloat for f64 {
    type UINT = u64;

    fn from_bits(bits: Self::UINT) -> Self {
        f64::from_bits(bits)
    }

    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    fn from_u16(value: u16) -> Self::UINT {
        value as u64
    }
}

impl ALPRDFloat for f32 {
    type UINT = u32;

    fn from_bits(bits: Self::UINT) -> Self {
        f32::from_bits(bits)
    }

    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    fn from_u16(value: u16) -> Self::UINT {
        value as u32
    }
}

/// Encoder for ALP-RD ("real doubles") values.
///
/// The encoder calculates its parameters from a single sample of floating-point values,
/// and then can be applied to many vectors.
///
/// ALP-RD uses the algorithm outlined in Section 3.4 of the paper. The crux of it is that the front
/// (most significant) bits of many double vectors tend to be  the same, i.e. most doubles in a
/// vector often use the same exponent and front bits. Compression proceeds by finding the best
/// prefix of up to 16 bits that can be collapsed into a dictionary of
/// up to 8 elements. Each double can then be broken into the front/left `L` bits, which neatly
/// bit-packs down to 1-3 bits per element (depending on the actual dictionary size).
/// The remaining `R` bits naturally bit-pack.
///
/// In the ideal case, this scheme allows us to store a sequence of doubles in 49 bits-per-value.
///
/// Our implementation draws on the MIT-licensed [C++ implementation] provided by the original authors.
///
/// [C++ implementation]: https://github.com/cwida/ALP/blob/main/include/alp/rd.hpp
pub struct RDEncoder {
    right_bit_width: u8,
    codes: Vec<u16>,
}

/// The "cut" ALP-RD vector.
///
/// ALP-RD will take a vector of input floating point numbers into a left-parts and a right-parts,
/// split along a cut-point. The left and right values are held separately.
pub struct Split<F, U> {
    /// Bit-packed left parts
    left_parts: Vec<u16>,

    /// Exceptions for the left_parts that could not be dictionary encoded.
    left_exceptions: Exceptions<u16>,

    /// Dictionary for encoding the `left_parts`.
    left_dict: Vec<u16>,

    /// Bit-packed right parts
    right_parts: Vec<U>,

    /// Bit-width for the `right_parts` component.
    right_parts_bit_width: u8,

    phantom_data: PhantomData<F>,
}

impl<T, U> Split<T, U> {
    /// Consume the parts of the result.
    pub fn into_parts(self) -> (Vec<u16>, Vec<u16>, Exceptions<u16>, Vec<U>, u8) {
        (
            self.left_parts,
            self.left_dict,
            self.left_exceptions,
            self.right_parts,
            self.right_parts_bit_width,
        )
    }

    /// Access the bit-width for just the right parts.
    pub fn right_parts_bit_width(&self) -> u8 {
        self.right_parts_bit_width
    }
}

impl<F, U> Split<F, U>
where
    F: ALPRDFloat<UINT = U>,
{
    /// Decode back into a vector of the floating point type.
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
    /// Build a new encoder from a sample of doubles.
    pub fn new<T>(sample: &[T]) -> Self
    where
        T: ALPRDFloat,
    {
        let dictionary = find_best_dictionary::<T>(sample);

        let mut codes = vec![0; dictionary.dictionary.len()];
        dictionary.dictionary.into_iter().for_each(|(bits, code)| {
            // write the reverse mapping into the codes vector.
            codes[code as usize] = bits
        });

        Self {
            right_bit_width: dictionary.right_bit_width,
            codes,
        }
    }

    /// Encode the floating point values into a result type.
    pub fn split<T>(&self, doubles: &[T]) -> Split<T, T::UINT>
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

        // mask for right-parts
        let right_mask = T::UINT::one().shl(self.right_bit_width as _) - T::UINT::one();
        let max_code = self.codes.len() - 1;
        let left_bit_width = bit_width!(max_code);

        for v in doubles.iter().copied() {
            right_parts.push(T::to_bits(v) & right_mask);
            left_parts.push(<T as ALPRDFloat>::to_u16(
                T::to_bits(v).shr(self.right_bit_width as _),
            ));
        }

        // dict-encode the left-parts, keeping track of exceptions
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

        // Bit-pack the dict-encoded left_parts
        // let left_parts = fastlanes_pack(&left_parts, left_bit_width);
        // // Bit-pack the right_parts
        // let right_parts = fastlanes_pack(&right_parts, self.right_bit_width as _);

        // TODO(aduffy): pack the exception_pos.
        let left_exceptions = Exceptions::new(exception_values, exception_pos);

        Split {
            left_parts,
            left_dict: self.codes.clone(),
            left_exceptions,
            right_parts,
            right_parts_bit_width: self.right_bit_width,
            phantom_data: PhantomData,
        }
    }
}

/// Decode a vector of ALP-RD encoded values back into their original floating point format.
///
/// # Panics
///
/// The function panics if the provided `left_parts` and `right_parts` differ in length.
///
/// The function panics if the provided `exc_pos` and `exceptions` differ in length.
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

    let mut dict = Vec::with_capacity(left_parts_dict.len());
    dict.extend_from_slice(left_parts_dict);

    let mut left_parts_decoded: Vec<T::UINT> = Vec::with_capacity(left_parts.len());

    // Decode with bit-packing and dict unpacking.
    for code in left_parts {
        left_parts_decoded.push(<T as ALPRDFloat>::from_u16(dict[*code as usize]));
    }

    // Apply the exception patches to left_parts
    for (pos, val) in exc_pos.iter().zip(exceptions.iter()) {
        left_parts_decoded[*pos as usize] = <T as ALPRDFloat>::from_u16(*val);
    }

    // recombine the left-and-right parts, adjusting by the right_bit_width.
    left_parts_decoded
        .into_iter()
        .zip(right_parts.iter().copied())
        .map(|(left, right)| T::from_bits((left << (right_bit_width as usize)) | right))
        .collect()
}

/// Find the best "cut point" for a set of floating point values such that we can
/// cast them all to the relevant value instead.
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T]) -> ALPRDDictionary {
    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();

    for p in 1..=16 {
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

/// Build dictionary of the leftmost bits.
fn build_left_parts_dictionary<T: ALPRDFloat>(
    samples: &[T],
    right_bw: u8,
    max_dict_size: u8,
) -> (ALPRDDictionary, usize) {
    assert!(
        right_bw >= (T::BITS - CUT_LIMIT) as _,
        "left-parts must be <= 16 bits"
    );

    // Count the number of occurrences of each left bit pattern
    let mut counts = HashMap::new();
    samples
        .iter()
        .copied()
        .map(|v| <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(right_bw as _)))
        .for_each(|item| *counts.entry(item).or_default() += 1);

    // Sorted counts: sort by negative count so that heavy hitters sort first.
    let mut sorted_bit_counts: Vec<(u16, usize)> = counts.into_iter().collect();
    sorted_bit_counts.sort_by_key(|(_, count)| count.wrapping_neg());

    // Assign the most-frequently occurring left-bits as dictionary codes, up to `dict_size`...
    let mut dictionary = HashMap::with_capacity(max_dict_size as _);
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
    let left_bw = bit_width!(max_code) as u8;

    (
        ALPRDDictionary {
            dictionary,
            right_bit_width: right_bw,
            left_bit_width: left_bw,
        },
        exception_count,
    )
}

/// Estimate the bits-per-value when using these compression settings.
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
    dictionary: HashMap<u16, u16>,
    /// The (compressed) left bit width. This is after bit-packing the dictionary codes.
    left_bit_width: u8,
    /// The right bit width. This is the bit-packed width of each of the "real double" values.
    right_bit_width: u8,
}

#[cfg(test)]
mod test {
    use crate::RDEncoder;

    #[test]
    fn test_encode_decode() {
        let values = vec![1.12345f64, 2.34567f64, 3.45678f64];

        let encoder = RDEncoder::new(&values);

        let split = encoder.split(&values);
        let decoded = split.decode();
        assert_eq!(decoded, values);
    }
}
