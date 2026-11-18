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
    MatricizeContract()

Contract by matricizing both operands and multiplying the matrices. The default algorithm for
`AbstractArray` operands.
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
