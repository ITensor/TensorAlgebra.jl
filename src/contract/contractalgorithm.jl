"""
    ContractAlgorithm

Supertype of contraction algorithms, passed as the `alg` keyword of [`contract`](@ref) and
related functions. A new algorithm subtypes `ContractAlgorithm` and implements
[`contractpermconjadd!`](@ref) with the algorithm as its first argument.

`ContractAlgorithm(alg)` converts `alg` to a `ContractAlgorithm`. With TensorOperations loaded,
`ContractAlgorithm(backend)` wraps a TensorOperations backend in a
[`TensorOperationsContract`](@ref).
"""
abstract type ContractAlgorithm <: AbstractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

"""
    MatricizeContract()

Contract by matricizing both operands, multiplying the matrices and unfolding the product into
the destination. It is the default for `AbstractArray` operands. It goes through
[`matricizeconj`](@ref) and [`unmatricizeadd!`](@ref), so an array type supports it by
implementing the matricization hooks.
"""
struct MatricizeContract <: ContractAlgorithm end

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
