#![feature(float_next_up_down)]

//! This crate contains an implementation of the floating point compression algorithm from the
//! paper ["ALP: Adaptive Lossless floating-Point Compression"][paper] by Afroozeh et al.
//!
//! The compressor has two variants, classic ALP which is well-suited for data that does not use
//! the full precision, and "real doubles", values that do.
//!
//! Classic ALP will return small integers, and it is meant to be cascaded with other integer
//! compression techniques such as bit-packing and frame-of-reference encoding. Combined this allows
//! for significant compression on the order of what you can get for integer values.
//!
//! ALP-RD is generally terminal, and in the ideal case it can represent an f64 is just 49 bits,
//! though generally it is closer to 54 bits per value or ~12.5% compression.
//!
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf

pub use alp::*;
pub use alp_rd::*;

mod alp;
mod alp_rd;

/// A sparse vector containing exceptions to the encoding process.
///
/// When either of the ALP variants encounters values it is unable to compress, they are stored
/// in here using the actual encoding offsets instead.
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
    pub fn new(values: Vec<T>, positions: Vec<u64>) -> Self {
        Self { values, positions }
    }

    /// Apply the exceptions to the given array.
    pub fn apply(&self, vec: &mut [T]) {
        self.values.iter().zip(self.positions.iter())
            .for_each(|(value, pos)| vec[*pos as usize] = *value);
    }
}

#[cfg(test)]
mod test {
    use crate::Exceptions;

    #[test]
    fn test_apply_exceptions() {
        let exceptions = Exceptions::new(
            vec![0u8; 4],
            vec![1, 2, 3],
        );

        let mut values = vec![10; 4];
        exceptions.apply(&mut values);

        assert_eq!(values, vec![10, 0, 0, 0]);
    }
}