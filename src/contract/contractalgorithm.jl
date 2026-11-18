"""
    ContractAlgorithm

Supertype of contraction algorithms, passed as the `alg` keyword of [`contract`](@ref) and
related functions. A new algorithm implements [`contractpermconjadd!`](@ref) with the
algorithm as its first argument.
"""
abstract type ContractAlgorithm <: AbstractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

"""
    MatricizeContract(; permute_alg = nothing, allocator = TensorAlgebra.DefaultAllocator())

Contract by matricizing both operands and multiplying the matrices. The default algorithm for
`AbstractArray` operands. `permute_alg` is the [`PermuteAlgorithm`](@ref) of the matricizations
(`nothing` selects the default), and `allocator` provides the temporary matrices.
"""
struct MatricizeContract{PermuteAlg, Allocator} <: ContractAlgorithm
    permute_alg::PermuteAlg
    allocator::Allocator
end
function MatricizeContract(; permute_alg = nothing, allocator = DefaultAllocator())
    return MatricizeContract(permute_alg, allocator)
end
allocator(algorithm::MatricizeContract) = algorithm.allocator

"""
    TensorOperationsContract(; backend = nothing, allocator = nothing)

Contract using TensorOperations, with `backend` selecting the contraction kernel and
`allocator` the allocator for temporary tensors (e.g. `TensorOperations.ManualAllocator()`).
A `nothing` field uses TensorOperations' default. Only usable with TensorOperations loaded.
"""
Base.@kwdef struct TensorOperationsContract{Backend, Allocator} <:
    ContractAlgorithm
    backend::Backend = nothing
    allocator::Allocator = nothing
end
