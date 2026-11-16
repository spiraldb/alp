# Benchmarking

ALP uses microbenchmarks to measure the encode and decode kernels. CodSpeed runs them in CI and
tracks the results.

## Microbenchmarks

The microbenchmarks use Criterion, through `codspeed-criterion-compat`, and live in the `benches/`
directory. Run all of them, one target, or the cases that match a filter:

```bash
cargo bench
cargo bench --bench alp_decode
cargo bench --bench alp_decode -- into_16k
```

### Give inputs by reference from the setup closure

Use `iter_batched_ref` for each benchmark. The setup closure gives the inputs, and the measured
closure takes them by reference:

```rust
let encoded = make_encoded();

group.bench_function("alloc", |b| {
    b.iter_batched_ref(
        || (encoded.as_slice(), exponents),
        |(encoded, exponents)| alp::decode::<f64>(encoded, *exponents),
        BatchSize::SmallInput,
    );
});
```

This one pattern obeys three rules:

- **Setup is not measured.** Construct shared fixtures before the call to `bench_function`, and
  return references to them from the setup closure. Criterion calls the setup closure outside the
  measured region.
- **The compiler cannot remove the work.** The compiler can fold or remove work when it knows the
  inputs. Criterion black-boxes the value that the setup closure returns and the value that the
  measured closure returns. Thus calls to `black_box` are not necessary.
- **Destruction is not measured.** Criterion drops the inputs and the result after the measurement
  stops. Return the result from the closure. Do not discard it inside the closure.

Do not use `iter` with captured values. Captured values are not black-boxed, and `iter` drops the
result inside the measured region.

Do the one-time initialization of process state before the first benchmark. Lazy initialization
inside a measured iteration can be larger than the operation.

### Make output buffers in the setup closure

When the operation writes into a buffer of the caller, make the buffer in the setup closure. The
allocation and the release of the buffer are then outside the measured region:

```rust
b.iter_batched_ref(
    || (encoded.as_slice(), exponents, vec![0.0f64; LEN]),
    |(encoded, exponents, output)| alp::decode_into::<f64>(encoded, *exponents, output),
    BatchSize::LargeInput,
);
```

Use `BatchSize::LargeInput` when the setup closure allocates. In a local run, Criterion keeps the
inputs and the results of a full batch until the batch ends, and a large batch of buffers uses much
memory. CodSpeed ignores the batch size.

### Keep the results of a manual loop

Inside a manual loop, push each result into a vector from the setup closure. The compiler cannot
discard the earlier iterations, and the results are dropped after the measurement stops:

```rust
b.iter_batched_ref(
    || (&encoder, values.as_slice(), Vec::with_capacity(LEN / CHUNK)),
    |(encoder, values, splits)| {
        for chunk in values.chunks(CHUNK) {
            splits.push(encoder.split(chunk));
        }
    },
    BatchSize::SmallInput,
);
```

Other values that the closure drops are part of the measurement. Keep those drops only when they are
part of the cost that you measure.

### Choose reproducible inputs

Use a seeded generator for random data. Random data without a seed can change the measured work
between runs although the code stays the same. `benches/alp_rd.rs` has a small linear congruential
generator, `Lcg`, which keeps the benchmarks repeatable and free of more dependencies.

Choose inputs that do useful work. An empty input or an input with nothing to patch can measure
mostly the benchmark harness. For very small operations, repeat a fixed quantity of work and keep
each result. Set the input size in bytes when you compare `f32` with `f64`.

Keep each iteration below 1 ms so that the suite stays practical to run. This is the cost of one
call of the closure, not the total time of the benchmark binary. Examine the `HEAD` value for each
iteration in the CodSpeed report before you merge a new benchmark. If a benchmark is above this
limit, decrease the input size, divide the workload into parameterized cases, or keep the benchmark
local. The module comment in `benches/alp_rd.rs` shows how to record the reason for an input size.

For CPU kernels, keep the working set in the L2 cache of the runner unless memory bandwidth is the
intended measurement.

### Parameterize workloads

Use a benchmark group with `bench_with_input` and `BenchmarkId` for different workloads. Keep the
parameter list in one place:

```rust
let mut group = c.benchmark_group("encoder_new");

for len in [64usize, 1_024, 4_096, 16_384] {
    let values = log_normal(len);

    group.bench_with_input(BenchmarkId::new("log_normal", len), &values, |b, values| {
        b.iter_batched_ref(
            || values.as_slice(),
            |values| RDEncoder::new(values),
            BatchSize::SmallInput,
        );
    });
}

group.finish();
```

Use a generic function to run the same cases for `f32` and `f64`, as `bench_float` does in
`benches/alp_encode.rs`. Set `group.throughput` so that a local run also reports elements for each
second.

A benchmark name identifies its history in CodSpeed. If you rename a benchmark, CodSpeed starts a
new history for it.

## CodSpeed

The `bench` job in
[`.github/workflows/ci.yml`](https://github.com/spiraldb/alp/blob/develop/.github/workflows/ci.yml)
runs the benchmarks on each push to `develop` and on each pull request, which includes pull requests
from forks. The results are on [CodSpeed](https://app.codspeed.io/spiraldb/alp).

The job uses the [CPU simulation](https://codspeed.io/docs/instruments/cpu) instrument. CodSpeed
calls the closure five times as a warm-up, then estimates cycles from the instruction trace of one
more call. The estimate includes cache and memory access costs. The Criterion settings
`sample_size`, `measurement_time`, and `warm_up_time` have no effect in CI.

To do the same build locally, install `cargo-codspeed` and run:

```bash
cargo codspeed build --all-features
cargo codspeed run
```

Without the CodSpeed runner, `cargo codspeed run` only makes sure that the benchmarks run. It does
not give measurements.

### Give short benchmarks sufficient measured work

Simulation has a fixed harness cost. One small allocation can take only tens of nanoseconds, so most
of its reported time can come from that cost. Give each iteration a minimum of a few microseconds of
useful work.

### Benchmarks share one process

The job sets `CODSPEED_PROCESS_ISOLATION` to `"false"`, so all the benchmarks of one target run in
one process. Allocation can depend on the process state that the earlier benchmarks leave. Thus, if
you add, remove, or move a benchmark, the results of the benchmarks that run after it can change.

Do not change this setting without a good reason. A change breaks the comparison with the earlier
history.

### Keep incompatible benchmarks out of automatic runs

Simulation does not include the time of system calls. Do not use it to track an operation that
depends on I/O or kernel calls.

Use the automatic reports to track the maintained ALP entry points. Keep third-party kernels and
frozen implementations for local comparisons only. An unexplained change in their automatic results
does not identify a change in the maintained ALP code.

`cargo codspeed build` sets the `codspeed` cfg. Use `#[cfg(not(codspeed))]` for a benchmark that
cannot give useful CodSpeed measurements:

```rust
fn bench(c: &mut Criterion) {
    bench_maintained(c);

    #[cfg(not(codspeed))]
    bench_reference(c);
}
```

The lints in `Cargo.toml` deny an unknown cfg name. Before the first use, add `codspeed` to the
known names:

```toml
[workspace.lints.rust]
unexpected_cfgs = { level = "deny", check-cfg = ['cfg(codspeed)'] }
```

When only one input case is unreliable, remove that case from the parameter list for CodSpeed
builds. Write the reason adjacent to the exclusion and keep the supporting measurements in the pull
request. Keep the CI commands independent of individual benchmark names.

### Investigate reports on unchanged code

Make sure that you know the exact base and head commits before you attribute a change. An old
comparison base can include unrelated changes. Then examine the flame graph in the CodSpeed report
for changes in the measured work or in allocation.

Simulation removes clock noise, but the instruction trace can change on unchanged code:

- If many later benchmarks move after you add or remove an earlier benchmark, examine the shared
  allocator state.
- If one benchmark moves after an unrelated source change, examine code placement. A different
  layout of the binary can change alignment and inlining.

Neither pattern is proof of the cause.

To measure the noise, run the same executable again in new processes. To measure the sensitivity to
code placement, compare independent builds. Keep the commands and the environment the same. One
repeatable executable does not show that unrelated source changes cannot change its measurements.

Record the base and head commits, the commands, and the environment with the results.

If a case stays noisy after you correct its setup, exclude it from automatic runs and put the
evidence in the pull request. Before you put it back, show that it is repeatable and that a
deliberate increase in the measured work stays visible. A stable result is not useful if the
measurement does not show a real regression. Do not increase the regression threshold to hide noise.
