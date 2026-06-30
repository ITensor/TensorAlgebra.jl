using BenchmarkTools: warmup
using Test: @test, @testset

# Run the benchmark suite once, so it cannot fall out of sync with the API without
# CI noticing. This is a smoke test, not a regression check (for that, compare
# revisions with `benchpkg`, see `benchmark/README.md`).
include(joinpath(@__DIR__, "..", "benchmark", "benchmarks.jl"))

@testset "benchmarks (smoke run)" begin
    @test (warmup(SUITE); true)
end
