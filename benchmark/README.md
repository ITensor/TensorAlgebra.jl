# TensorAlgebra benchmarks

A [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl) suite for
TensorAlgebra. The benchmarks live in the `SUITE` global defined in
`benchmarks.jl`, covering the core contraction entry points (`contract`,
`contract!`) across a range of dimensions. Small dimensions track the fixed
per-call label bookkeeping, large ones the BLAS call, so a regression in either
regime shows up.

The suite is also run once as a smoke test in the package tests (`test_benchmarks.jl`),
so it cannot silently fall out of sync with the API.

## Running the suite

```bash
julia --project=benchmark -e '
    include("benchmark/benchmarks.jl")
    using BenchmarkTools
    display(run(SUITE; verbose = true))'
```

## Comparing two revisions

With [`AirspeedVelocity.jl`](https://github.com/MilesCranmer/AirspeedVelocity.jl)
installed in a shared environment, compare the working tree against `main`:

```bash
benchpkg TensorAlgebra --rev=dirty,main -o benchmark/results/
benchpkgtable TensorAlgebra --rev=dirty,main -i benchmark/results/
```

Timings on shared or CI machines are noisy. Compare allocation counts and relative
numbers rather than absolute times, and run on a quiet machine when absolute
timings matter.
