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
//! # Classic ALP
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
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf

pub use alp::*;
pub use alp_rd::*;

mod alp;
mod alp_rd;

/// A sparse vector containing exceptions to the encoding process.
///
/// When either of the ALP variants encounters values it is unable to compress, they are stored
/// here using the actual encoding offsets instead.
///
/// Indices should be stored bit-packed, so that they can be accessed that way.
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
