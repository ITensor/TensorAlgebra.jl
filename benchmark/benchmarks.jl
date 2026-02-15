using BenchmarkTools
using TensorAlgebra

SUITE = BenchmarkGroup()

const CONTRACTIONS_PATH = joinpath(@__DIR__, "benchmark_specs", "randomTCs.dat")

include("contractions.jl")

# Contraction benchmarks
# ----------------------
contraction_suite = SUITE["contractions"] = BenchmarkGroup()

Ts = (Float64, ComplexF64)
algs = (TensorAlgebra.Matricize(),)

for alg in algs
    alg_suite = contraction_suite[alg] = BenchmarkGroup()
    for T in Ts
        alg_suite[T] = BenchmarkGroup()

        for (i, line) in enumerate(eachline(CONTRACTIONS_PATH))
            alg_suite[T][i] = generate_contract_benchmark(line; T, alg)
        end
    end
end
