//! This crate contains an implementation of the floating-point compression algorithm from the
//! paper ["ALP: Adaptive Lossless floating-Point Compression"][paper] by Afroozeh et al.
//!
//! The compressor has two variants: classic ALP, which is well-suited for data that does not use
//! the full precision, and "real doubles", for values that do.
//!
//! Classic ALP will return small integers, and it is meant to be cascaded with other integer
//! compression techniques such as bit-packing and frame-of-reference encoding. Combined, this
//! allows for significant compression on the order of what you can get for integer values.
//!
//! ALP-RD is generally terminal, and in the ideal case it can represent an f64 in just 49 bits,
//! though generally it is closer to 54 bits per value or ~12.5% compression.
//!
//! # ALP
//!
//! [`encode`] picks the best exponents for the input (unless they are given), and returns the
//! encoded integers alongside the positions and values of the exceptions that do not round-trip.
//! Exceptional slots in the encoded output hold a fill value, so that they stay in range for
//! downstream integer compression. Values are encoded in chunks of [`ENCODE_CHUNK_SIZE`], and one
//! offset per chunk is returned, so a consumer can find the exceptions of a chunk without scanning
//! the whole patch index.
//!
//! ```
//! let values = vec![1.234f64, 5.678, 9.0];
//! let (exponents, encoded, patch_indices, patch_values, chunk_offsets) =
//!     alp::encode(&values, None);
//!
//! assert_eq!(encoded, vec![1234, 5678, 9000]);
//! assert!(patch_indices.is_empty() && patch_values.is_empty());
//! assert_eq!(chunk_offsets, vec![0]);
//! assert_eq!(alp::decode::<f64>(&encoded, exponents), values);
//! ```
//!
//! Decoding is usually done in place, over the encoded buffer itself
//! ([`decode_slice_inplace`]). If you plan to do that, encode with [`encode_into`] so the values
//! land in a buffer you own from the start, rather than in a `Vec` you would have to adopt:
//!
//! ```
//! # let values = vec![1.234f64, 5.678, 9.0];
//! let mut encoded: Vec<i64> = Vec::with_capacity(values.len());
//! let (mut patch_indices, mut patch_values, mut chunk_offsets) = (vec![], vec![], vec![]);
//!
//! let exponents = alp::encode_into(
//!     &values,
//!     None,
//!     &mut encoded.spare_capacity_mut()[..values.len()],
//!     &mut patch_indices,
//!     &mut patch_values,
//!     &mut chunk_offsets,
//! );
//! // SAFETY: `encode_into` initializes one element per value.
//! unsafe { encoded.set_len(values.len()) };
//!
//! alp::decode_slice_inplace::<f64>(&mut encoded, exponents);
//! ```
//!
//! # ALP-RD
//!
//! [`RDEncoder`] derives a dictionary of the most common front bits from a sample, and then splits
//! values into dictionary-encoded left parts and bit-packable right parts. Left parts that are not
//! in the dictionary are stored as exceptions.
//!
//! ```
//! let values = vec![0.1f64, 0.2, 3e100];
//! let encoder = alp::RDEncoder::new(&values[0..2]);
//!
//! let split = encoder.split(&values);
//! assert_eq!(split.left_exceptions().positions(), &[2]);
//! assert_eq!(split.decode(), values);
//! ```
//!
//! # Bit-packing
//!
//! [fastlanes] packs 1024 values at a time, which is also [`ENCODE_CHUNK_SIZE`], so encode and pack
//! at the same granularity: a chunk at a time. A shorter chunk is padded out to 1024 values, so
//! packing a handful of values saves nothing.
//!
//! ## Bit-packing ALP
//!
//! Classic ALP's encoded integers are signed and sit wherever the data does, so packing them starts
//! with a frame of reference: subtract the minimum, pack the differences, keep the minimum. Patched
//! slots hold a fill value taken from the data itself, so they do not widen that frame.
//!
//! | Component | Where | Count | Pack at |
//! | --- | --- | --- | --- |
//! | encoded values | [`encode`]'s 2nd result | one per value | [`bit_width`] of `max - min`, after subtracting `min` |
//! | patch positions | [`encode`]'s 3rd result | one per exception | as is, or 10 bits for a 1024-value chunk |
//! | patch values | [`encode`]'s 4th result | one per exception | as is: they are the original floats, at full precision |
//! | chunk offsets | [`encode`]'s 5th result | one per [`ENCODE_CHUNK_SIZE`] | as is |
//! | exponents | [`encode`]'s 1st result | two bytes per column | as is |
//!
//! ```
//! # use fastlanes::BitPacking;
//! # fn pack<T: BitPacking>(chunk: &[T; 1024], width: usize) -> Vec<T> {
//! #     let mut packed = vec![T::zero(); 1024 * width / (size_of::<T>() * 8)];
//! #     unsafe { BitPacking::unchecked_pack(width, chunk, &mut packed) };
//! #     packed
//! # }
//! # fn unpack<T: BitPacking>(packed: &[T], width: usize) -> Vec<T> {
//! #     let mut chunk = vec![T::zero(); 1024];
//! #     unsafe { BitPacking::unchecked_unpack(width, packed, &mut chunk) };
//! #     chunk
//! # }
//! // Two decimal places, which is what classic ALP is for.
//! let mut values: Vec<f64> = (0..1024).map(|i| 100.0 + (i % 500) as f64 / 100.0).collect();
//! values[7] = 1.0 / 3.0; // Does not round-trip, so it becomes a patch.
//!
//! let (exponents, encoded, patch_positions, patch_values, _chunk_offsets) =
//!     alp::encode(&values, None);
//!
//! // Frame of reference, then pack the differences.
//! let min = encoded.iter().copied().min().expect("a non-empty chunk");
//! let max = encoded.iter().copied().max().expect("a non-empty chunk");
//! let width = alp::bit_width(max.wrapping_sub(min) as u64) as usize;
//!
//! let shifted: Vec<u64> = encoded.iter().map(|v| v.wrapping_sub(min) as u64).collect();
//! let shifted: [u64; 1024] = shifted.try_into().expect("one full chunk");
//! let packed = pack(&shifted, width);
//!
//! // 1,152 bytes where the input took 8,192, or 9 bits per value: the encoded integers run from
//! // 10000 to 10499, and the patched slot holds a fill value inside that range.
//! assert!(size_of_val(packed.as_slice()) < size_of_val(values.as_slice()));
//!
//! // Undo the frame, decode, and write the patches over the slots that did not round-trip.
//! let unshifted: Vec<i64> = unpack(&packed, width)
//!     .into_iter()
//!     .map(|v| (v as i64).wrapping_add(min))
//!     .collect();
//!
//! let mut decoded = alp::decode::<f64>(&unshifted, exponents);
//! for (&position, &value) in patch_positions.iter().zip(&patch_values) {
//!     decoded[position as usize] = value;
//! }
//! assert_eq!(decoded, values);
//! ```
//!
//! ## Bit-packing ALP-RD
//!
//! A [`Split`] is five arrays. The two per-value ones are what bit-packing is for:
//!
//! | Component | Where | Count | Pack at |
//! | --- | --- | --- | --- |
//! | left parts (dictionary codes) | [`Split::left_parts`] | one per value | [`Split::left_parts_bit_width`]: 1 to 3 bits |
//! | right parts | [`Split::right_parts`] | one per value | [`Split::right_parts_bit_width`]: 48 to 63 bits for an `f64` |
//! | dictionary | [`Split::left_dict`] | at most [`MAX_DICT_SIZE`] `u16`s | as is — it belongs to the encoder, so it is shared by every chunk that encoder splits |
//! | exception patterns | [`Exceptions::values`] | one per exception | as is: at most 16 bits already |
//! | exception positions | [`Exceptions::positions`] | one per exception | as is, or 10 bits for a 1024-value chunk, if a chunk has enough exceptions to pay for a pack |
//!
//! ```
//! use fastlanes::BitPacking;
//!
//! /// Packs one 1024-value chunk down to `width` bits per value.
//! fn pack<T: BitPacking>(chunk: &[T; 1024], width: usize) -> Vec<T> {
//!     let mut packed = vec![T::zero(); 1024 * width / (size_of::<T>() * 8)];
//!     // SAFETY: a full chunk in, exactly as many words out as 1024 values of `width` bits
//!     // occupy, and `width` is no wider than `T`.
//!     unsafe { BitPacking::unchecked_pack(width, chunk, &mut packed) };
//!     packed
//! }
//!
//! /// Unpacks a chunk `pack` produced.
//! fn unpack<T: BitPacking>(packed: &[T], width: usize) -> Vec<T> {
//!     let mut chunk = vec![T::zero(); 1024];
//!     // SAFETY: as above, in reverse.
//!     unsafe { BitPacking::unchecked_unpack(width, packed, &mut chunk) };
//!     chunk
//! }
//!
//! // "Real doubles": these use the full mantissa, so classic ALP has nothing to work with.
//! let values: Vec<f64> = (0..1024).map(|i| (i as f64 + 0.5).sqrt()).collect();
//!
//! let encoder = alp::RDEncoder::new(&values);
//! let split = encoder.split(&values);
//!
//! let (left_width, right_width) = (
//!     split.left_parts_bit_width() as usize,
//!     split.right_parts_bit_width() as usize,
//! );
//! let left: [u16; 1024] = split.left_parts().try_into().expect("one full chunk");
//! let right: [u64; 1024] = split.right_parts().try_into().expect("one full chunk");
//!
//! let packed_left = pack(&left, left_width);
//! let packed_right = pack(&right, right_width);
//!
//! // Those two arrays are the whole payload: 6,912 bytes where the input took 8,192, or 54 bits
//! // per value — 51 for a right part, 3 for a dictionary code.
//! let nbytes = size_of_val(packed_left.as_slice()) + size_of_val(packed_right.as_slice());
//! assert!(nbytes < size_of_val(values.as_slice()));
//!
//! // Decoding takes the two parts unpacked, the encoder's dictionary, and the exceptions as they
//! // came.
//! let decoded = alp::alp_rd_decode::<f64>(
//!     &unpack(&packed_left, left_width),
//!     split.left_dict(),
//!     split.right_parts_bit_width(),
//!     &unpack(&packed_right, right_width),
//!     split.left_exceptions().positions(),
//!     split.left_exceptions().values(),
//! );
//! assert_eq!(decoded, values);
//!
//! // Random access survives packing: one value costs one unpacking operation per part, plus a look
//! // at the exceptions, whose positions come out ascending.
//! let position = 512;
//! let left = match split.left_exceptions().positions().binary_search(&(position as u64)) {
//!     Ok(exception) => split.left_exceptions().values()[exception],
//!     Err(_) => {
//!         let code = unsafe { u16::unchecked_unpack_single(left_width, &packed_left, position) };
//!         split.left_dict()[code as usize]
//!     }
//! };
//! let right = unsafe { u64::unchecked_unpack_single(right_width, &packed_right, position) };
//! assert_eq!(
//!     f64::from_bits((u64::from(left) << right_width) | right),
//!     values[position],
//! );
//! ```
//!
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf
//! [fastlanes]: https://docs.rs/fastlanes
//! [vortex]: https://docs.rs/vortex

pub use alp::*;
pub use alp_rd::*;

mod alp;
mod alp_rd;

// Runs the README's examples as doctests, so they cannot drift from the API they demonstrate.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

/// A sparse vector containing exceptions to the encoding process.
///
/// When either of the ALP variants encounters values it is unable to compress, they are stored
/// here using the actual encoding offsets instead.
///
/// Positions are indices into the array that was encoded, in ascending order, so they can be
/// searched for the exception belonging to a value. [The section on bit-packing](crate#bit-packing)
/// covers how to store them.
pub struct Exceptions<T> {
    values: Vec<T>,
    positions: Vec<u64>,
}

impl<T> Exceptions<T>
where
    T: Copy,
{
    /// Creates a set of exceptions from the values and the positions they belong at.
    ///
    /// # Panics
    ///
    /// Panics if `values` and `positions` have different lengths.
    pub fn new(values: Vec<T>, positions: Vec<u64>) -> Self {
        assert_eq!(
            values.len(),
            positions.len(),
            "Exceptions: values.len != positions.len"
        );
        Self { values, positions }
    }

    /// Returns the exceptional values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the positions of the exceptional values.
    #[inline]
    pub fn positions(&self) -> &[u64] {
        &self.positions
    }

    /// Returns the number of exceptions.
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if there are no exceptions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Consumes the exceptions, returning the values and their positions.
    pub fn into_parts(self) -> (Vec<T>, Vec<u64>) {
        (self.values, self.positions)
    }

    /// Applies the exceptions to the given array.
    #[inline]
    pub fn apply(&self, vec: &mut [T]) {
        self.values
            .iter()
            .zip(self.positions.iter())
            .for_each(|(value, pos)| vec[*pos as usize] = *value);
    }
}

#[cfg(test)]
mod test {
    use crate::Exceptions;

    #[test]
    fn test_apply_exceptions() {
        let exceptions = Exceptions::new(vec![0u8; 3], vec![1, 2, 3]);

        let mut values = vec![10; 4];
        exceptions.apply(&mut values);

        assert_eq!(values, vec![10, 0, 0, 0]);
        assert_eq!(exceptions.len(), 3);
        assert_eq!(exceptions.positions(), &[1, 2, 3]);
        assert_eq!(exceptions.values(), &[0, 0, 0]);
    }
}
