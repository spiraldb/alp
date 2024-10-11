# ALP: Adaptive Lossless floating-Point

As modern data and analytics workloads have shifted from SQL to general-purpose programming
languages such as Python, the amount of floating point data has grown massively. It is a
problem for modern database systems to effectively compress this data without loss of precision,
while preserving desirable traits such as random access and auto-vectorization.

In 2023, [Afroozeh et al.](https://dl.acm.org/doi/pdf/10.1145/3626717) published ALP,
a response to these issues. The code was written in [C++](https://github.com/cwida/ALP) and integrated
into DuckDB. To ease the integration into other tools, we present a Rust implementation of both variants
of ALP (ALP and ALP for "real doubles").

