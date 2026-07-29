//! Benchmarks for the ALP-RD encoder.
//!
//! Two things are worth guarding here.
//!
//! `RDEncoder::new` rescans the whole sample for every candidate cut point, so its cost currently
//! scales with the length of the sample. The `encoder_new` group sweeps sample sizes from tiny to
//! several million so that a change which caps that work — by subsampling, say — shows up as the
//! curve flattening. The small sizes matter as much as the large ones: an implementation that pays
//! a fixed cost per cut point, such as clearing or scanning a 64K-entry table, looks fine on
//! millions of values and is an order of magnitude slower on the few thousand a caller typically
//! samples.
//!
//! `RDEncoder::split` runs once per chunk over the whole dataset, so `split_chunks` measures it at
//! the 1024-value chunk size a columnar layout would use.

use alp::{RDEncoder, alp_rd_decode};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// A small linear congruential generator, to keep the benchmarks dependency-free and repeatable.
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
}

/// Log-normal values spanning many exponents: a high-cardinality stress case for the dictionary
/// search, and the shape a lot of scientific float data takes.
fn log_normal(len: usize) -> Vec<f64> {
    let mut rng = Lcg::new();
    (0..len)
        .map(|_| (rng.next_unit() * 6.0 - 1.0).exp() * 1000.0)
        .collect()
}

/// Values from a single narrow range, so the left parts collapse to a handful of patterns. This is
/// the case ALP-RD is designed for and where exceptions should be rare.
fn narrow_range(len: usize) -> Vec<f64> {
    let mut rng = Lcg::new();
    (0..len).map(|_| 1.0 + rng.next_unit()).collect()
}

/// Values of two very different magnitudes, interleaved. Periodic input is where a striding
/// sampler aliases onto one phase and picks a bad cut point; this keeps an eye on the cost of
/// sampling contiguous runs instead.
fn interleaved(len: usize) -> Vec<f64> {
    let mut rng = Lcg::new();
    (0..len)
        .map(|i| {
            let magnitude = if i % 3 == 0 { 1e-8 } else { 1e12 };
            magnitude * (1.0 + rng.next_unit())
        })
        .collect()
}

/// `RDEncoder::new` should cost about the same regardless of how much sample it is handed.
fn bench_encoder_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_new");

    for len in [64usize, 1_024, 4_096, 65_536, 1 << 20, 1 << 22] {
        let values = log_normal(len);
        group.bench_with_input(BenchmarkId::new("log_normal", len), &values, |b, values| {
            b.iter(|| RDEncoder::new(black_box(values.as_slice())));
        });
    }

    for len in [64usize, 1_024, 65_536, 1 << 20] {
        let values = narrow_range(len);
        group.bench_with_input(
            BenchmarkId::new("narrow_range", len),
            &values,
            |b, values| b.iter(|| RDEncoder::new(black_box(values.as_slice()))),
        );
    }

    for len in [65_536usize, 1 << 20] {
        let values = interleaved(len);
        group.bench_with_input(
            BenchmarkId::new("interleaved", len),
            &values,
            |b, values| b.iter(|| RDEncoder::new(black_box(values.as_slice()))),
        );
    }

    group.finish();
}

/// `split` over the whole dataset in the chunk size a columnar layout would use.
fn bench_split_chunks(c: &mut Criterion) {
    const CHUNK: usize = 1024;
    const LEN: usize = 1 << 20;

    let mut group = c.benchmark_group("split_chunks");
    group.throughput(Throughput::Elements(LEN as u64));

    for (name, values) in [
        ("log_normal", log_normal(LEN)),
        ("narrow_range", narrow_range(LEN)),
    ] {
        let encoder = RDEncoder::new(&values);
        group.bench_function(name, |b| {
            b.iter(|| {
                for chunk in values.chunks(CHUNK) {
                    black_box(encoder.split(black_box(chunk)));
                }
            });
        });
    }

    group.finish();
}

/// Decoding, split across the unpatched fast path and the patched one.
fn bench_decode(c: &mut Criterion) {
    const LEN: usize = 1 << 20;

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(LEN as u64));

    // Trained on the same data, so almost nothing excepts: the unpatched fast path.
    let clean = narrow_range(LEN);
    // Trained on a narrow prefix, so the wider values except: the patched path.
    let patched = log_normal(LEN);

    for (name, values, encoder) in [
        (
            "unpatched",
            &clean,
            RDEncoder::new(&clean[..clean.len().min(4096)]),
        ),
        ("patched", &patched, RDEncoder::new(&patched[..64])),
    ] {
        let split = encoder.split(values);
        let (left_parts, left_dict, exceptions, right_parts, right_bit_width) = split.into_parts();
        group.bench_function(name, |b| {
            b.iter(|| {
                alp_rd_decode::<f64>(
                    black_box(&left_parts),
                    black_box(&left_dict),
                    right_bit_width,
                    black_box(&right_parts),
                    exceptions.positions(),
                    exceptions.values(),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encoder_new, bench_split_chunks, bench_decode);
criterion_main!(benches);
