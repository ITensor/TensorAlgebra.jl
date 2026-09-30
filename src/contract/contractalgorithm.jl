abstract type ContractAlgorithm <: AbstractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

struct MatricizeContract <: ContractAlgorithm end

"""
    TensorOperationsContract(; backend = nothing, allocator = nothing, temporary = false)

Contract using TensorOperations, with `backend` selecting the contraction kernel and
`allocator` the allocator for temporary tensors (e.g. `TensorOperations.ManualAllocator()`).
A `nothing` field uses TensorOperations' default. Only usable with TensorOperations loaded.
`temporary = true` marks outputs allocated by `contract` as temporary for `allocator`, to be
released with `TensorOperations.tensorfree!`.
"""
Base.@kwdef struct TensorOperationsContract{Backend, Allocator} <:
    ContractAlgorithm
    backend::Backend = nothing
    allocator::Allocator = nothing
    temporary::Bool = false
end
function TensorOperationsContract(backend, allocator)
    return TensorOperationsContract(backend, allocator, false)
end
