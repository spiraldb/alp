//! ALP decoding benchmarks for allocating and caller-provided output buffers.

#![allow(missing_docs)]

use alp::{ALPFloat, Exponents};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_float<F: ALPFloat>(c: &mut Criterion, name: &str) {
    const LEN: usize = 1024;
    let exponents = Exponents { e: 3, f: 0 };
    let values: Vec<F> = (0..LEN)
        .map(|i| {
            F::from((i * 7919) % 100000).expect("small integers fit both float types") * F::IF10[3]
        })
        .collect();
    let (_, encoded, patches, _, _) = alp::encode::<F>(&values, Some(exponents));
    assert!(patches.is_empty());
    assert!(alp::decode::<F>(&encoded, exponents) == values);

    let mut group = c.benchmark_group(format!("alp_decode_{name}"));
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("alloc", |b| {
        b.iter(|| alp::decode::<F>(black_box(&encoded), black_box(exponents)));
    });
    let mut output = vec![F::zero(); LEN];
    group.bench_function("into", |b| {
        b.iter(|| {
            alp::decode_into::<F>(
                black_box(&encoded),
                black_box(exponents),
                black_box(&mut output),
            );
        });
    });
    let scaled = Exponents { e: 6, f: 3 };
    group.bench_function("into_scaled", |b| {
        b.iter(|| {
            alp::decode_into::<F>(
                black_box(&encoded),
                black_box(scaled),
                black_box(&mut output),
            );
        });
    });
    let large: Vec<F::ALPInt> = encoded.iter().copied().cycle().take(16384).collect();
    let mut large_output = vec![F::zero(); large.len()];
    group.throughput(Throughput::Elements(large.len() as u64));
    for (case, exp) in [("into_16k", exponents), ("into_scaled_16k", scaled)] {
        group.bench_function(case, |b| {
            b.iter(|| {
                alp::decode_into::<F>(
                    black_box(&large),
                    black_box(exp),
                    black_box(&mut large_output),
                );
            });
        });
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_float::<f64>(c, "f64");
    bench_float::<f32>(c, "f32");
}

criterion_group!(benches, bench);
criterion_main!(benches);
