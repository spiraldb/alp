# ALP: Adaptive Lossless floating-Point

[![crates.io](https://img.shields.io/crates/v/alp.svg)](https://crates.io/crates/alp)
[![docs.rs](https://img.shields.io/docsrs/alp)](https://docs.rs/alp)
[![CI](https://github.com/spiraldb/alp/actions/workflows/ci.yml/badge.svg)](https://github.com/spiraldb/alp/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/alp.svg)](LICENSE)

As modern data and analytics workloads have shifted from SQL to general-purpose programming
languages such as Python, the amount of floating-point data has grown massively. It is a
problem for modern database systems to effectively compress this data without loss of precision,
while preserving desirable traits such as random access and auto-vectorization.

In 2023, [Afroozeh et al.](https://dl.acm.org/doi/pdf/10.1145/3626717) published ALP,
a response to these issues. The code was written in [C++](https://github.com/cwida/ALP) and integrated
into DuckDB. To ease the integration into other tools, we present a Rust implementation of both variants
of ALP (ALP and ALP for "real doubles").

```sh
cargo add alp
```

## Which variant

| | Classic ALP | ALP-RD |
| --- | --- | --- |
| Suits | values with few significant decimal digits: prices, sensor readings, rounded measurements | "real doubles" that use the full mantissa: coordinates, embeddings, physical constants |
| How | multiplies by a power of ten and rounds to an integer, keeping the values that do not round-trip as exceptions | cuts every value in two, dictionary-encodes the leading bits and keeps the rest verbatim |
| Yields | integers spanning the range of the input, to compress further with frame-of-reference and bit-packing | 1 to 3 bits per left part, plus the cut point's worth of right part: 49 bits per `f64` at best, ~54 typically |

Both are lossless, and both are meant to be run a chunk of 1024 values at a time — the granularity
bit-packing works at.

## Classic ALP

```rust
let values = vec![1.234f64, 5.678, 9.0];
let (exponents, encoded, patch_positions, patch_values, chunk_offsets) = alp::encode(&values, None);

// 1.234 * 10^3, and nothing that failed to round-trip.
assert_eq!(encoded, vec![1234, 5678, 9000]);
assert!(patch_positions.is_empty() && patch_values.is_empty());

assert_eq!(alp::decode::<f64>(&encoded, exponents), values);
```

## ALP-RD

```rust
let values = vec![0.1f64, 0.2, 3e100];

// The encoder learns its dictionary from a sample, then splits any number of chunks with it.
let encoder = alp::RDEncoder::new(&values[0..2]);
let split = encoder.split(&values);

// 3e100 has an exponent the sample never showed, so its left part is an exception.
assert_eq!(split.left_exceptions().positions(), &[2]);
assert_eq!(split.decode(), values);
```

## Bit-packing

Result of ALP and ALP-RD requires bitpacking to actually save space by themselves both variants
only prepare the data for further compression. The [bit-packing section of the docs][bit-packing]
walks through both variants end to end.

## Benchmarks

```sh
cargo bench
```

Results are tracked on [CodSpeed](https://app.codspeed.io/spiraldb/alp).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). The ALP-RD implementation draws on the
MIT-licensed [C++ implementation](https://github.com/cwida/ALP) by the paper's authors.

[bit-packing]: https://docs.rs/alp/latest/alp/#bit-packing
[fastlanes]: https://github.com/spiraldb/fastlanes
[Vortex]: https://github.com/spiraldb/vortex
