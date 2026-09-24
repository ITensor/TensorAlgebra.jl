abstract type ContractAlgorithm <: AbstractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

struct MatricizeContract{LeftStyle, RightStyle, OutputStyle} <: ContractAlgorithm
    left_matricize_style::LeftStyle
    right_matricize_style::RightStyle
    output_matricize_style::OutputStyle
end
function MatricizeContract(matricize_style)
    return MatricizeContract(matricize_style, matricize_style, matricize_style)
end
MatricizeContract() = MatricizeContract(ReshapeMatricize())

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
