use fastlanes::BitPacking;

/// A bit-packed vector with a fixed bit-width.
///
/// Logically, a `PackedVec<T>` is a vector containing values of type `T`.
///
/// Physically, the values are stored bit-packed down to a given width. They can be individually
/// accessed by performing single unpacking operation.
pub struct PackedVec<T> {
    values: Vec<T>,
    packed_width: usize,
}

/// Bitpack a slice of primitives down to the given width using the FastLanes layout.
pub(crate) fn fastlanes_pack<T: BitPacking>(
    array: &[T],
    bit_width: usize,
) -> Vec<T> {
    if bit_width == 0 {
        return Vec::new();
    }

    // How many fastlanes vectors we will process.
    let num_chunks = (array.len() + 1023) / 1024;
    let num_full_chunks = array.len() / 1024;
    let packed_len = 128 * bit_width / size_of::<T>();
    // packed_len says how many values of size T we're going to include.
    // 1024 * bit_width / 8 == the number of bytes we're going to get.
    // then we divide by the size of T to get the number of elements.

    // Allocate a result byte array.
    let mut output = Vec::<T>::with_capacity(num_chunks * packed_len);

    // Loop over all but the last chunk.
    (0..num_full_chunks).for_each(|i| {
        let start_elem = i * 1024;

        output.reserve(packed_len);
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &array[start_elem..][..1024],
                &mut output[output_len..][..packed_len],
            );
        };
    });

    // Pad the last chunk with zeros to a full 1024 elements.
    if num_chunks != num_full_chunks {
        let last_chunk_size = array.len() % 1024;
        let mut last_chunk: [T; 1024] = [T::zero(); 1024];
        last_chunk[..last_chunk_size].copy_from_slice(&array[array.len() - last_chunk_size..]);

        output.reserve(packed_len);
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &last_chunk,
                &mut output[output_len..][..packed_len],
            );
        };
    }

    output
}


pub(crate) fn fastlanes_unpack<T: BitPacking>(
    packed: &[T],
    bit_width: usize,
    offset: usize,
    length: usize,
) -> Vec<T> {
    if bit_width == 0 {
        return vec![T::zero(); length];
    }

    // How many fastlanes vectors we will process.
    // Packed array might not start at 0 when the array is sliced. Offset is guaranteed to be < 1024.
    let num_chunks = (offset + length + 1023) / 1024;
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    assert_eq!(
        packed.len(),
        num_chunks * elems_per_chunk,
        "Invalid packed length: got {}, expected {}",
        packed.len(),
        num_chunks * elems_per_chunk
    );

    // Allocate a result vector.
    let mut output = Vec::with_capacity(num_chunks * 1024 - offset);

    // Handle first chunk if offset is non 0. We have to decode the chunk and skip first offset elements
    let first_full_chunk = if offset != 0 {
        let chunk: &[T] = &packed[0..elems_per_chunk];
        let mut decoded = [T::zero(); 1024];
        unsafe { BitPacking::unchecked_unpack(bit_width, chunk, &mut decoded) };
        output.extend_from_slice(&decoded[offset..]);
        1
    } else {
        0
    };

    // Loop over all the chunks.
    (first_full_chunk..num_chunks).for_each(|i| {
        let chunk: &[T] = &packed[i * elems_per_chunk..][0..elems_per_chunk];
        unsafe {
            let output_len = output.len();
            output.set_len(output_len + 1024);
            BitPacking::unchecked_unpack(bit_width, chunk, &mut output[output_len..][0..1024]);
        }
    });

    // The final chunk may have had padding
    output.truncate(length);

    // For small vectors, the overhead of rounding up is more noticable.
    // Shrink to fit may or may not reallocate depending on the implementation.
    // But for very small vectors, the reallocation is cheap enough even if it does happen.
    if output.len() < 1024 {
        output.shrink_to_fit();
    }

    assert_eq!(
        output.len(),
        length,
        "Expected unpacked array to be of length {} but got {}",
        length,
        output.len()
    );
    output
}


// TODO: add fastlanes_unpack_dict that fuses dictionary lookup with unpacking.


#[cfg(test)]
mod test {
    use crate::alp_rd::bitpack::{fastlanes_pack, fastlanes_unpack};

    #[test]
    fn test_pack_unpack() {
        let values: Vec<u8> = (0u8..16u8).collect();
        let packed = fastlanes_pack(&values, 4);
        let unpacked = fastlanes_unpack(&packed, 4, 0, values.len());
        assert_eq!(unpacked, values);
    }
}
