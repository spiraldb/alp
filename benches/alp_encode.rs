//! ALP encoding benchmarks separating exponent search, allocation, and exception handling.

#![allow(missing_docs)]

use alp::{ALPFloat, Exponents};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::mem::MaybeUninit;

fn bench_float<F: ALPFloat>(c: &mut Criterion, name: &str) {
    const LEN: usize = 1024;
    let exponents = Exponents { e: 3, f: 0 };
    let values: Vec<F> = (0..LEN)
        .map(|i| {
            F::from((i * 7919) % 100000).expect("small integers fit both float types") * F::IF10[3]
        })
        .collect();
    let mut perturbed = values.clone();
    for value in perturbed.iter_mut().skip(64).step_by(32) {
        *value =
            *value + F::from(0.0000123456789).expect("finite perturbation fits both float types");
    }
    let mut dense = values.clone();
    for value in dense.iter_mut().step_by(2) {
        *value =
            *value + F::from(0.0000123456789).expect("finite perturbation fits both float types");
    }
    let all_patched = vec![F::nan(); LEN];
    let mut group = c.benchmark_group(format!("alp_encode_{name}"));
    group.throughput(Throughput::Elements(LEN as u64));
    for (case, input) in [
        ("clean", &values),
        ("patched", &perturbed),
        ("dense", &dense),
        ("all_patched", &all_patched),
    ] {
        let (_, encoded, positions, exceptions, _) = alp::encode::<F>(input, Some(exponents));
        assert_eq!(positions.is_empty(), case == "clean");
        let mut decoded = alp::decode::<F>(&encoded, exponents);
        for (&position, &value) in positions.iter().zip(&exceptions) {
            decoded[position as usize] = value;
        }
        assert!(
            decoded
                .iter()
                .zip(input)
                .all(|(&actual, &expected)| actual.is_eq(expected))
        );

        group.bench_function(format!("{case}/alloc"), |b| {
            b.iter(|| alp::encode::<F>(black_box(input), Some(black_box(exponents))));
        });
        let mut encoded = vec![MaybeUninit::<F::ALPInt>::uninit(); LEN];
        let mut positions = Vec::with_capacity(LEN);
        let mut exceptions = Vec::with_capacity(LEN);
        let mut offsets = Vec::with_capacity(1);
        group.bench_function(format!("{case}/into"), |b| {
            b.iter(|| {
                positions.clear();
                exceptions.clear();
                offsets.clear();
                alp::encode_into::<F>(
                    black_box(input),
                    Some(black_box(exponents)),
                    black_box(&mut encoded),
                    black_box(&mut positions),
                    black_box(&mut exceptions),
                    black_box(&mut offsets),
                );
            });
        });
    }
    group.finish();

    let wide: Vec<F> = (0..LEN)
        .map(|i| {
            F::from((i % 61) as i32 - 30)
                .expect("small exponents fit both float types")
                .exp()
        })
        .collect();
    let mut group = c.benchmark_group(format!("alp_exponent_search_{name}"));
    for (case, input) in [
        ("clean", &values),
        ("patched", &perturbed),
        ("wide", &wide),
        ("all_patched", &all_patched),
    ] {
        group.bench_function(case, |b| {
            b.iter(|| alp::find_best_exponents(black_box(input)));
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
