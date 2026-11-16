//! ALP decoding benchmarks for allocating and caller-provided output buffers.

#![allow(missing_docs)]

use alp::{ALPFloat, Exponents};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

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
    // The setup closure gives each input by reference. Criterion black-boxes the inputs and drops
    // the decoded vector after the measurement stops.
    group.bench_function("alloc", |b| {
        b.iter_batched_ref(
            || (encoded.as_slice(), exponents),
            |(encoded, exponents)| alp::decode::<F>(encoded, *exponents),
            BatchSize::SmallInput,
        );
    });

    // The setup closure makes the output buffer, so its allocation is not part of the measurement.
    let scaled = Exponents { e: 6, f: 3 };
    for (case, exp) in [("into", exponents), ("into_scaled", scaled)] {
        group.bench_function(case, |b| {
            b.iter_batched_ref(
                || (encoded.as_slice(), exp, vec![F::zero(); LEN]),
                |(encoded, exp, output)| alp::decode_into::<F>(encoded, *exp, output),
                BatchSize::LargeInput,
            );
        });
    }

    let large: Vec<F::ALPInt> = encoded.iter().copied().cycle().take(16384).collect();
    group.throughput(Throughput::Elements(large.len() as u64));
    for (case, exp) in [("into_16k", exponents), ("into_scaled_16k", scaled)] {
        group.bench_function(case, |b| {
            b.iter_batched_ref(
                || (large.as_slice(), exp, vec![F::zero(); large.len()]),
                |(large, exp, output)| alp::decode_into::<F>(large, *exp, output),
                BatchSize::LargeInput,
            );
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
