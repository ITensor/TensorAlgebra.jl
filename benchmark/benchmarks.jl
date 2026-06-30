using BenchmarkTools: @benchmarkable, BenchmarkGroup
using TensorAlgebra: contract, contract!

# Benchmarks of the core contraction entry points across a range of dimensions.
# Small dimensions are dominated by the fixed per-call label bookkeeping, large
# ones by the BLAS call, so a regression in either regime is visible. The `SUITE`
# global is the entry point `AirspeedVelocity.jl`'s `benchpkg` expects.

const SUITE = BenchmarkGroup()

const DIMS = (4, 16, 64)

contract_suite = SUITE["contract"] = BenchmarkGroup()

# Matrix multiply, one shared index: C[i,k] = A[i,j] B[j,k].
matmul = contract_suite["matmul"] = BenchmarkGroup()
for d in DIMS
    matmul[d] = @benchmarkable(
        contract(A, (:i, :j), B, (:j, :k)),
        setup = (A = randn($d, $d); B = randn($d, $d)),
    )
end

# Rank-3 contraction over two shared indices: C[i,l] = A[i,j,k] B[k,j,l].
rank3 = contract_suite["rank3"] = BenchmarkGroup()
for d in (4, 16)
    rank3[d] = @benchmarkable(
        contract(A, (:i, :j, :k), B, (:k, :j, :l)),
        setup = (A = randn($d, $d, $d); B = randn($d, $d, $d)),
    )
end

# Full contraction to a scalar: c = A[i,j] B[i,j].
scalar = contract_suite["scalar"] = BenchmarkGroup()
for d in DIMS
    scalar[d] = @benchmarkable(
        contract(A, (:i, :j), B, (:i, :j)),
        setup = (A = randn($d, $d); B = randn($d, $d)),
    )
end

# In-place matrix multiply into a preallocated destination.
contract!_suite = SUITE["contract!"] = BenchmarkGroup()
matmul! = contract!_suite["matmul"] = BenchmarkGroup()
for d in DIMS
    matmul![d] = @benchmarkable(
        contract!(C, (:i, :k), A, (:i, :j), B, (:j, :k)),
        setup = (A = randn($d, $d); B = randn($d, $d); C = zeros($d, $d)),
    )
end
