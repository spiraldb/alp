use itertools::Itertools;
use num_traits::{CheckedSub, Float, PrimInt, ToPrimitive};
use std::fmt::{Display, Formatter};
use std::mem::{MaybeUninit, size_of, transmute, transmute_copy};

const SAMPLE_SIZE: usize = 32;

/// The number of values encoded per chunk.
///
/// [`ALPFloat::encode`] processes its input in chunks of this size and emits one entry per chunk
/// into the returned chunk offsets, allowing consumers to find the patches belonging to a chunk
/// without scanning the whole patch index.
pub const ENCODE_CHUNK_SIZE: usize = 1024;

/// The pair of powers of ten a value is scaled by.
///
/// Encoding computes `round(v * 10^e / 10^f)` and decoding undoes it, so the encoded integer keeps
/// `e - f` decimal digits of the original value. Splitting the scale in two is what lets ALP hold
/// values that need a large `10^e` to become integral without inflating the integers it emits.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Exponents {
    /// Power of ten the value is multiplied by before it is rounded to an integer.
    pub e: u8,
    /// Power of ten the value is divided by before it is rounded, undoing part of `e`.
    pub f: u8,
}

impl Display for Exponents {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "e: {}, f: {}", self.e, self.f)
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Encodes an array of floating-point values, finding the best exponents if they are not provided.
///
/// Returns the exponents, the encoded values, the positions and values of the exceptions, and the
/// offset into the exception positions at which each [`ENCODE_CHUNK_SIZE`] chunk begins.
#[inline]
#[allow(clippy::type_complexity)]
pub fn encode<F: ALPFloat>(
    values: &[F],
    exponents: Option<Exponents>,
) -> (Exponents, Vec<F::ALPInt>, Vec<u64>, Vec<F>, Vec<u64>) {
    F::encode(values, exponents)
}

/// Encodes an array of floating-point values into a buffer owned by the caller.
///
/// See [`ALPFloat::encode_into`]. Prefer this over [`encode`] when the encoded values should live
/// in your own buffer, for instance because you intend to decode them in place later.
#[inline]
pub fn encode_into<F: ALPFloat>(
    values: &[F],
    exponents: Option<Exponents>,
    encoded: &mut [MaybeUninit<F::ALPInt>],
    patch_indices: &mut Vec<u64>,
    patch_values: &mut Vec<F>,
    chunk_offsets: &mut Vec<u64>,
) -> Exponents {
    F::encode_into(
        values,
        exponents,
        encoded,
        patch_indices,
        patch_values,
        chunk_offsets,
    )
}

/// Finds the best exponent pair for a given set of values, trying to minimize the number of
/// exceptions.
#[inline]
pub fn find_best_exponents<F: ALPFloat>(values: &[F]) -> Exponents {
    F::find_best_exponents(values)
}

/// Encodes a single value, returning `None` if it does not round-trip under `exponents`.
#[inline]
pub fn encode_single<F: ALPFloat>(value: F, exponents: Exponents) -> Option<F::ALPInt> {
    F::encode_single(value, exponents)
}

/// Decodes an integer value to its matching floating-point representation given the same exponents.
#[inline]
pub fn decode_single<F: ALPFloat>(encoded: F::ALPInt, exponents: Exponents) -> F {
    F::decode_single(encoded, exponents)
}

/// Decodes a slice of encoded values into a newly allocated vector.
#[inline]
pub fn decode<F: ALPFloat>(encoded: &[F::ALPInt], exponents: Exponents) -> Vec<F> {
    F::decode(encoded, exponents)
}

/// Decodes a slice of encoded values into the provided output slice.
///
/// # Panics
///
/// Panics if `encoded` and `output` have different lengths.
#[inline]
pub fn decode_into<F: ALPFloat>(encoded: &[F::ALPInt], exponents: Exponents, output: &mut [F]) {
    F::decode_into(encoded, exponents, output)
}

/// Decodes a slice of encoded values in-place, reinterpreting it as a slice of floats.
#[inline]
pub fn decode_slice_inplace<F: ALPFloat>(encoded: &mut [F::ALPInt], exponents: Exponents) {
    F::decode_slice_inplace(encoded, exponents)
}

/// Encodes a single value without checking that it round-trips.
///
/// The returned value might decode to a different value than the one passed in. Use
/// [`encode_single`] for the checked version.
#[inline]
pub fn encode_single_unchecked<F: ALPFloat>(value: F, exponents: Exponents) -> F::ALPInt {
    F::encode_single_unchecked(value, exponents)
}

/// Widens a running `(min, max)` bound to include `value`, seeding it on the first value.
fn update_bounds<I: Ord + Copy>(bounds: &mut Option<(I, I)>, value: I) {
    *bounds = Some(bounds.map_or((value, value), |(min, max)| {
        (min.min(value), max.max(value))
    }));
}

/// Main trait for classic-ALP encodable floating-point numbers.
///
/// We limit this to the IEEE 754 single-precision (`f32`) and double-precision
/// (`f64`) floating-point types.
pub trait ALPFloat: private::Sealed + Float + Display + 'static {
    /// The signed integer type of the same width, which values of this type encode to.
    type ALPInt: PrimInt + Display + ToPrimitive;

    /// Number of fractional (mantissa) bits the type holds.
    const FRACTIONAL_BITS: u8;
    /// Exclusive upper bound on the exponents [`Self::find_best_exponents`] tries, set by the
    /// largest power of ten [`Self::ALPInt`] can hold.
    const MAX_EXPONENT: u8;
    /// The value [`Self::fast_round`] adds and subtracts to round without branching: the smallest
    /// magnitude above which the type holds no fractional part at all.
    const SWEET: Self;
    /// Powers of ten, `10^i` at index `i`.
    const F10: &'static [Self];
    /// Inverse powers of ten, `10^-i` at index `i`.
    const IF10: &'static [Self];

    /// Rounds to the nearest floating integer by shifting in and out of the low precision range.
    #[inline]
    fn fast_round(self) -> Self {
        (self + Self::SWEET) - Self::SWEET
    }

    /// Casts the primitive float to the target integer type, equivalent to calling `as`.
    fn as_int(self) -> Self::ALPInt;

    /// Converts from the integer type back to the float type using `as`.
    fn from_int(n: Self::ALPInt) -> Self;

    /// Compares bit-wise, so that `NaN`s and signed zeros are distinct values.
    fn is_eq(self, other: Self) -> bool;

    /// Finds the exponent pair with the smallest estimated encoded size.
    ///
    /// Inputs longer than a few dozen values are sampled rather than scanned in full, since the
    /// number of decimal digits a column uses is a property of the column rather than of any one
    /// value.
    fn find_best_exponents(values: &[Self]) -> Exponents {
        let mut best_exp = Exponents { e: 0, f: 0 };
        let mut best_nbytes: usize = usize::MAX;

        let sample = (values.len() > SAMPLE_SIZE).then(|| {
            values
                .iter()
                .step_by(values.len() / SAMPLE_SIZE)
                .cloned()
                .collect_vec()
        });
        let sample = sample.as_deref().unwrap_or(values);

        for e in (0..Self::MAX_EXPONENT).rev() {
            for f in 0..e {
                let exp = Exponents { e, f };
                let size = Self::estimate_encoded_size_for_exponents(sample, exp);
                if size < best_nbytes {
                    best_nbytes = size;
                    best_exp = exp;
                } else if size == best_nbytes && e - f < best_exp.e - best_exp.f {
                    best_exp = exp;
                }
            }
        }

        best_exp
    }

    /// Estimates the encoded size of `values` under `exponents`, matching a full
    /// [`Self::encode`] plus [`Self::estimate_encoded_size`] but without the per-candidate
    /// allocations.
    fn estimate_encoded_size_for_exponents(values: &[Self], exponents: Exponents) -> usize {
        // `kept` is the (min, max) over values that round-trip exactly (kept inline by `encode`);
        // `all` is the (min, max) over every encoded value. `encode` fills patched slots in-range,
        // so its emitted range is `kept`, except with all values patched (no fill), where `all`
        // wins.
        let mut kept: Option<(Self::ALPInt, Self::ALPInt)> = None;
        let mut all: Option<(Self::ALPInt, Self::ALPInt)> = None;
        let mut patch_count = 0usize;

        for &value in values {
            let encoded = Self::encode_single_unchecked(value, exponents);
            update_bounds(&mut all, encoded);
            if Self::decode_single(encoded, exponents).is_eq(value) {
                update_bounds(&mut kept, encoded);
            } else {
                patch_count += 1;
            }
        }

        let range = if patch_count == values.len() {
            all
        } else {
            kept
        };

        let bits_per_encoded = range
            .and_then(|(min, max)| max.checked_sub(&min))
            .and_then(|range_size| range_size.to_u64())
            .and_then(|range_size| {
                range_size
                    .checked_ilog2()
                    .map(|bits| (bits + 1) as usize)
                    .or(Some(0))
            })
            .unwrap_or(size_of::<Self::ALPInt>() * 8);

        let encoded_bytes = (values.len() * bits_per_encoded).div_ceil(8);
        // Each patch is a value plus a position. In practice, patch positions are in
        // [0, u16::MAX] because of how we chunk.
        let patch_bytes = patch_count * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    /// Estimates the bytes `encoded` and `patches` occupy once cascaded, assuming the encoded
    /// values get a frame of reference and are bit-packed, and the patches are stored as they are.
    #[inline]
    fn estimate_encoded_size(encoded: &[Self::ALPInt], patches: &[Self]) -> usize {
        let bits_per_encoded = encoded
            .iter()
            .minmax()
            .into_option()
            // Estimate the bits per encoded value, assuming frame-of-reference plus
            // bit-packing without patches.
            .and_then(|(min, max)| max.checked_sub(min))
            .and_then(|range_size: <Self as ALPFloat>::ALPInt| range_size.to_u64())
            .and_then(|range_size| {
                range_size
                    .checked_ilog2()
                    .map(|bits| (bits + 1) as usize)
                    .or(Some(0))
            })
            .unwrap_or(size_of::<Self::ALPInt>() * 8);

        let encoded_bytes = (encoded.len() * bits_per_encoded).div_ceil(8);
        // Each patch is a value plus a position. In practice, patch positions are in
        // [0, u16::MAX] because of how we chunk.
        let patch_bytes = patches.len() * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    /// Encodes `values`, finding the best exponents if they are not provided.
    ///
    /// Returns the exponents, the encoded values, the positions and values of the exceptions, and
    /// the offset into the exception positions at which each [`ENCODE_CHUNK_SIZE`] chunk begins.
    ///
    /// Use [`Self::encode_into`] if the encoded values should land in a buffer you own, which is
    /// what you want if you intend to decode them in place later.
    #[allow(clippy::type_complexity)]
    fn encode(
        values: &[Self],
        exponents: Option<Exponents>,
    ) -> (Exponents, Vec<Self::ALPInt>, Vec<u64>, Vec<Self>, Vec<u64>) {
        let mut encoded_output = Vec::with_capacity(values.len());

        // Estimate capacity to be one patch per 32 values.
        let mut patch_indices = Vec::with_capacity(values.len() / 32);
        let mut patch_values = Vec::with_capacity(values.len() / 32);

        // There's exactly one offset per chunk.
        let mut chunk_offsets = Vec::with_capacity(values.len().div_ceil(ENCODE_CHUNK_SIZE));

        let exp = Self::encode_into(
            values,
            exponents,
            &mut encoded_output.spare_capacity_mut()[..values.len()],
            &mut patch_indices,
            &mut patch_values,
            &mut chunk_offsets,
        );

        // SAFETY: `encode_into` initializes exactly `values.len()` elements.
        unsafe { encoded_output.set_len(values.len()) };

        (
            exp,
            encoded_output,
            patch_indices,
            patch_values,
            chunk_offsets,
        )
    }

    /// Encodes `values` into an output buffer owned by the caller, returning the exponents used.
    ///
    /// `encoded` must hold exactly `values.len()` elements, all of which are initialized on
    /// return. The exception positions and values, plus one offset into the exception positions
    /// per [`ENCODE_CHUNK_SIZE`] chunk, are appended to the three vectors, which must start empty.
    ///
    /// This exists so that the encoded values can be written straight into a buffer the caller
    /// controls (an arena, an aligned buffer, or anything else it will later decode in place),
    /// rather than into a `Vec` that has to be adopted or copied afterwards.
    ///
    /// # Panics
    ///
    /// Panics if `encoded` is not the same length as `values`, or if any of `patch_indices`,
    /// `patch_values` or `chunk_offsets` is non-empty.
    fn encode_into(
        values: &[Self],
        exponents: Option<Exponents>,
        encoded: &mut [MaybeUninit<Self::ALPInt>],
        patch_indices: &mut Vec<u64>,
        patch_values: &mut Vec<Self>,
        chunk_offsets: &mut Vec<u64>,
    ) -> Exponents {
        assert_eq!(
            encoded.len(),
            values.len(),
            "encode_into: output buffer must hold exactly one element per value"
        );
        assert!(
            patch_indices.is_empty() && patch_values.is_empty() && chunk_offsets.is_empty(),
            "encode_into: patch and chunk offset outputs must start empty"
        );

        let exp = exponents.unwrap_or_else(|| Self::find_best_exponents(values));
        let mut fill_value: Option<Self::ALPInt> = None;

        // This is intentionally branchless.
        for (chunk_idx, chunk) in values.chunks(ENCODE_CHUNK_SIZE).enumerate() {
            chunk_offsets.push(patch_indices.len() as u64);
            encode_chunk_unchecked(
                chunk,
                chunk_idx * ENCODE_CHUNK_SIZE,
                exp,
                encoded,
                patch_indices,
                patch_values,
                &mut fill_value,
            );
        }

        exp
    }

    /// Encodes a single value, rounding up instead of to the nearest integer.
    #[inline]
    fn encode_above(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .ceil()
            .as_int()
    }

    /// Encodes a single value, rounding down instead of to the nearest integer.
    #[inline]
    fn encode_below(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .floor()
            .as_int()
    }

    /// Encodes a single value, returning `None` if it does not round-trip under `exponents`.
    #[inline]
    fn encode_single(value: Self, exponents: Exponents) -> Option<Self::ALPInt> {
        let encoded = Self::encode_single_unchecked(value, exponents);
        let decoded = Self::decode_single(encoded, exponents);
        if decoded.is_eq(value) {
            return Some(encoded);
        }
        None
    }

    /// Decodes `encoded` into a newly allocated vector.
    #[inline]
    fn decode(encoded: &[Self::ALPInt], exponents: Exponents) -> Vec<Self> {
        let mut values = Vec::with_capacity(encoded.len());
        for encoded in encoded {
            values.push(Self::decode_single(*encoded, exponents));
        }
        values
    }

    /// Decodes `encoded` into `output`.
    ///
    /// # Panics
    ///
    /// Panics if `encoded` and `output` have different lengths.
    #[inline]
    fn decode_into(encoded: &[Self::ALPInt], exponents: Exponents, output: &mut [Self]) {
        assert_eq!(encoded.len(), output.len());

        for i in 0..encoded.len() {
            output[i] = Self::decode_single(encoded[i], exponents)
        }
    }

    /// Decodes `encoded` in-place, reinterpreting the slice as floats of the same width.
    #[inline]
    fn decode_slice_inplace(encoded: &mut [Self::ALPInt], exponents: Exponents) {
        // SAFETY: `Self` and `Self::ALPInt` have the same size and alignment (f32/i32, f64/i64),
        // and every bit pattern is valid for both.
        let decoded: &mut [Self] = unsafe { transmute(encoded) };
        decoded.iter_mut().for_each(|v| {
            // SAFETY: as above, we're reading back the integer we have not yet overwritten.
            *v = Self::decode_single(
                unsafe { transmute_copy::<Self, Self::ALPInt>(v) },
                exponents,
            )
        })
    }

    /// Decodes a single value, undoing the scaling `exponents` applied.
    #[inline]
    fn decode_single(encoded: Self::ALPInt, exponents: Exponents) -> Self {
        Self::from_int(encoded) * Self::F10[exponents.f as usize] * Self::IF10[exponents.e as usize]
    }

    /// Encodes a single value without checking that it round-trips.
    ///
    /// The returned value might decode to a different value than the one passed in. Use
    /// [`Self::encode_single`] for the checked version.
    #[inline]
    fn encode_single_unchecked(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .fast_round()
            .as_int()
    }
}

/// Encodes one chunk into `encoded_output`, which covers the whole array: elements before
/// `num_prev_encoded` are already initialized, this chunk initializes the next `chunk.len()`.
fn encode_chunk_unchecked<T: ALPFloat>(
    chunk: &[T],
    num_prev_encoded: usize,
    exp: Exponents,
    encoded_output: &mut [MaybeUninit<T::ALPInt>],
    patch_indices: &mut Vec<u64>,
    patch_values: &mut Vec<T>,
    fill_value: &mut Option<T::ALPInt>,
) {
    let num_prev_patches = patch_indices.len();
    assert_eq!(patch_indices.len(), patch_values.len());
    let has_filled = fill_value.is_some();

    // Encode the chunk, counting the number of patches.
    let mut chunk_patch_count = 0;
    for (output, &v) in encoded_output[num_prev_encoded..][..chunk.len()]
        .iter_mut()
        .zip(chunk)
    {
        let encoded = T::encode_single_unchecked(v, exp);
        let decoded = T::decode_single(encoded, exp);
        let neq = !decoded.is_eq(v) as usize;
        chunk_patch_count += neq;
        output.write(encoded);
    }
    let chunk_patch_count = chunk_patch_count; // immutable hereafter

    let encoded_len = num_prev_encoded + chunk.len();
    // SAFETY: every element below `encoded_len` has been initialized, by this call for the values
    // of this chunk and by the preceding calls for everything before it.
    let encoded_output: &mut [T::ALPInt] = unsafe {
        std::slice::from_raw_parts_mut(encoded_output.as_mut_ptr().cast::<T::ALPInt>(), encoded_len)
    };

    if chunk_patch_count > 0 {
        // We need to gather the patches for this chunk. Preallocate space for them, plus one
        // because our loop may attempt to write one past the end.
        patch_indices.reserve(chunk_patch_count + 1);
        patch_values.reserve(chunk_patch_count + 1);

        // Record the patches in this chunk.
        let patch_indices_mut = patch_indices.spare_capacity_mut();
        let patch_values_mut = patch_values.spare_capacity_mut();
        let mut chunk_patch_index = 0;
        for i in num_prev_encoded..encoded_len {
            let decoded = T::decode_single(encoded_output[i], exp);
            // `write()` is only safe to call more than once because the values are primitive,
            // i.e. `Drop` is a no-op.
            patch_indices_mut[chunk_patch_index].write(i as u64);
            patch_values_mut[chunk_patch_index].write(chunk[i - num_prev_encoded]);
            chunk_patch_index += !decoded.is_eq(chunk[i - num_prev_encoded]) as usize;
        }
        assert_eq!(chunk_patch_index, chunk_patch_count);
        // SAFETY: we just initialized `chunk_patch_count` elements in the spare capacity of both
        // vectors, which we reserved above.
        unsafe {
            patch_indices.set_len(num_prev_patches + chunk_patch_count);
            patch_values.set_len(num_prev_patches + chunk_patch_count);
        }
    }

    // Find the first successfully encoded value, i.e. one that is not patched. This is our fill
    // value for missing values.
    if fill_value.is_none() && (num_prev_encoded + chunk_patch_count < encoded_len) {
        assert_eq!(num_prev_encoded, num_prev_patches);
        for i in num_prev_encoded..encoded_len {
            if i >= patch_indices.len() || patch_indices[i] != i as u64 {
                *fill_value = Some(encoded_output[i]);
                break;
            }
        }
    }

    // Replace the patched values in the encoded array with the fill value, for better downstream
    // compression.
    if let Some(fill_value) = fill_value {
        // Handle the edge case where the first N >= 1 chunks are all patches.
        let start_patch = if !has_filled { 0 } else { num_prev_patches };
        for patch_idx in &patch_indices[start_patch..] {
            encoded_output[*patch_idx as usize] = *fill_value;
        }
    }
}

impl ALPFloat for f32 {
    type ALPInt = i32;
    const FRACTIONAL_BITS: u8 = 23;
    const MAX_EXPONENT: u8 = 10;
    const SWEET: Self =
        (1 << Self::FRACTIONAL_BITS) as Self + (1 << (Self::FRACTIONAL_BITS - 1)) as Self;

    const F10: &'static [Self] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0, // 10^10
    ];
    const IF10: &'static [Self] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001, // 10^-10
    ];

    #[inline]
    fn as_int(self) -> Self::ALPInt {
        self as _
    }

    #[inline]
    fn from_int(n: Self::ALPInt) -> Self {
        n as _
    }

    #[inline]
    fn is_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl ALPFloat for f64 {
    type ALPInt = i64;
    const FRACTIONAL_BITS: u8 = 52;
    const MAX_EXPONENT: u8 = 18; // 10^18 is the maximum i64
    const SWEET: Self =
        (1u64 << Self::FRACTIONAL_BITS) as Self + (1u64 << (Self::FRACTIONAL_BITS - 1)) as Self;
    const F10: &'static [Self] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
        100000000000.0,
        1000000000000.0,
        10000000000000.0,
        100000000000000.0,
        1000000000000000.0,
        10000000000000000.0,
        100000000000000000.0,
        1000000000000000000.0,
        10000000000000000000.0,
        100000000000000000000.0,
        1000000000000000000000.0,
        10000000000000000000000.0,
        100000000000000000000000.0, // 10^23
    ];

    const IF10: &'static [Self] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001,
        0.00000000001,
        0.000000000001,
        0.0000000000001,
        0.00000000000001,
        0.000000000000001,
        0.0000000000000001,
        0.00000000000000001,
        0.000000000000000001,
        0.0000000000000000001,
        0.00000000000000000001,
        0.000000000000000000001,
        0.0000000000000000000001,
        0.00000000000000000000001, // 10^-23
    ];

    #[inline]
    fn as_int(self) -> Self::ALPInt {
        self as _
    }

    #[inline]
    fn from_int(n: Self::ALPInt) -> Self {
        n as _
    }

    #[inline]
    fn is_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The allocation-free estimate must match a full encode plus estimate for every candidate
    // exponent pair, so that `find_best_exponents` picks the same exponents and compression is
    // unchanged.
    fn check_estimate_matches<T: ALPFloat>(values: &[T]) {
        for e in 0..T::MAX_EXPONENT {
            for f in 0..e {
                let exp = Exponents { e, f };
                let lightweight = T::estimate_encoded_size_for_exponents(values, exp);
                let (_, encoded, _, patches, _) = T::encode(values, Some(exp));
                let full = T::estimate_encoded_size(&encoded, &patches);
                assert_eq!(
                    lightweight,
                    full,
                    "mismatch at e={e}, f={f}, len={}",
                    values.len()
                );
            }
        }
    }

    #[test]
    fn estimate_for_exponents_matches_full_encode() {
        // Clean 2-decimal values (mostly kept), repeating decimals (many patches), large
        // magnitudes, constants, and a single element.
        let mut f64s: Vec<f64> = (0..200).map(|i| i as f64 / 100.0).collect();
        f64s.extend((0..60).map(|i| i as f64 / 7.0));
        f64s.extend([1e17, -1e17, 0.0, 123.0]);
        check_estimate_matches(&f64s);
        check_estimate_matches::<f64>(&[123.456; 5]);
        check_estimate_matches::<f64>(&[42.0]);
        // Every value patches at every exponent -> exercises the all-patched branch.
        check_estimate_matches::<f64>(&[1.0 / 3.0; 8]);

        let mut f32s: Vec<f32> = (0..200).map(|i| i as f32 / 100.0).collect();
        f32s.extend((0..60).map(|i| i as f32 / 7.0));
        f32s.extend([1e9, -1e9, 0.0, 123.0]);
        check_estimate_matches(&f32s);
        check_estimate_matches::<f32>(&[1.0 / 3.0; 8]);
    }

    #[test]
    fn non_finite_numbers() {
        let original = vec![0.0f32, -0.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY];
        let (_, encoded, patch_idx, _, chunk_offsets) = encode(&original, None);

        assert_eq!(patch_idx, vec![1, 2, 3, 4]);
        assert_eq!(encoded, vec![0, 0, 0, 0, 0]);
        assert_eq!(chunk_offsets, vec![0]);
    }

    #[test]
    fn round_trip() {
        let original = vec![1.234f64; 1025];
        let (exponents, encoded, patch_idx, patch_values, chunk_offsets) = encode(&original, None);

        assert_eq!(exponents.e - exponents.f, 3);
        assert_eq!(encoded, vec![1234i64; 1025]);
        assert!(patch_idx.is_empty());
        assert!(patch_values.is_empty());
        assert_eq!(chunk_offsets, vec![0, 0]);

        assert_eq!(decode::<f64>(&encoded, exponents), original);
    }

    #[test]
    fn decode_matches_decode_single() {
        let original = vec![1.234f32, 5.678, -9.0, 0.001];
        let (exponents, encoded, ..) = encode(&original, None);

        let decoded = decode::<f32>(&encoded, exponents);
        assert_eq!(decoded, original);

        let mut into = vec![0f32; encoded.len()];
        decode_into::<f32>(&encoded, exponents, &mut into);
        assert_eq!(into, original);

        let mut inplace = encoded;
        decode_slice_inplace::<f32>(&mut inplace, exponents);
        // SAFETY: `i32` and `f32` have the same layout, and the slice now holds decoded floats.
        let inplace: &[f32] = unsafe { transmute(inplace.as_slice()) };
        assert_eq!(inplace, original);
    }

    #[test]
    fn chunk_offsets_are_per_chunk() {
        let mut values = vec![1.0f64; 3 * ENCODE_CHUNK_SIZE];
        values[1023] = std::f64::consts::PI;
        values[1024] = std::f64::consts::E;
        values[1025] = std::f64::consts::PI;

        let (_, _, patch_idx, _, chunk_offsets) = encode(&values, None);
        assert_eq!(patch_idx, vec![1023, 1024, 1025]);
        assert_eq!(chunk_offsets, vec![0, 1, 3]);
    }

    #[test]
    fn chunk_offsets_with_empty_chunks() {
        let mut values = vec![1.0f64; 3 * ENCODE_CHUNK_SIZE];
        values[0] = std::f64::consts::PI;
        values[2048] = std::f64::consts::E;

        let (_, _, patch_idx, _, chunk_offsets) = encode(&values, None);
        assert_eq!(patch_idx, vec![0, 2048]);
        assert_eq!(chunk_offsets, vec![0, 1, 1]);
    }

    #[test]
    fn encode_into_matches_encode() {
        // Also covers the multi-chunk path, where a chunk writes into the middle of the output.
        let mut values = vec![1.234f64; 2 * ENCODE_CHUNK_SIZE + 3];
        values[7] = std::f64::consts::PI;
        values[1500] = std::f64::consts::E;
        values[2050] = f64::NAN;

        let (exponents, encoded, patch_indices, patch_values, chunk_offsets) =
            encode(&values, None);

        let mut into_encoded: Vec<i64> = Vec::with_capacity(values.len());
        let mut into_patch_indices = Vec::new();
        let mut into_patch_values = Vec::new();
        let mut into_chunk_offsets = Vec::new();
        let into_exponents = encode_into(
            &values,
            None,
            &mut into_encoded.spare_capacity_mut()[..values.len()],
            &mut into_patch_indices,
            &mut into_patch_values,
            &mut into_chunk_offsets,
        );
        // SAFETY: `encode_into` initializes every element of the output.
        unsafe { into_encoded.set_len(values.len()) };

        assert_eq!(into_exponents, exponents);
        assert_eq!(into_encoded, encoded);
        assert_eq!(into_patch_indices, patch_indices);
        assert!(
            into_patch_values
                .iter()
                .zip(&patch_values)
                .all(|(a, b)| a.is_eq(*b))
        );
        assert_eq!(into_chunk_offsets, chunk_offsets);
    }

    #[test]
    #[should_panic(expected = "output buffer must hold exactly one element per value")]
    fn encode_into_rejects_wrong_sized_output() {
        let values = vec![1.234f64; 4];
        let mut encoded: Vec<i64> = Vec::with_capacity(2);
        encode_into(
            &values,
            None,
            &mut encoded.spare_capacity_mut()[..2],
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
    }

    #[test]
    fn encode_single_round_trips() {
        let exponents = Exponents { e: 16, f: 13 };
        assert_eq!(encode_single(1.234f64, exponents), Some(1234));
        assert_eq!(encode_single(std::f64::consts::PI, exponents), None);
        assert_eq!(decode_single::<f64>(1234, exponents), 1.234);
    }
}
