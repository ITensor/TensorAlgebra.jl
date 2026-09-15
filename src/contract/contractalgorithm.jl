abstract type ContractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

struct Matricize{LeftStyle, RightStyle, OutputStyle} <: ContractAlgorithm
    left_matricize_style::LeftStyle
    right_matricize_style::RightStyle
    output_matricize_style::OutputStyle
end
Matricize(matricize_style) = Matricize(matricize_style, matricize_style, matricize_style)
Matricize() = Matricize(ReshapeMatricize())

"""
    TensorOperationsAlgorithm(; backend = nothing, allocator = nothing)

Contract using TensorOperations, with `backend` selecting the contraction kernel and
`allocator` the allocator for temporary tensors (e.g. `TensorOperations.ManualAllocator()`).
A `nothing` field uses TensorOperations' default. Only usable with TensorOperations loaded.
"""
Base.@kwdef struct TensorOperationsAlgorithm{Backend, Allocator} <: ContractAlgorithm
    backend::Backend = nothing
    allocator::Allocator = nothing
end

# The contraction entry points collect trailing keywords and forward them here, so these accept
# `kwargs...` even though no `ContractAlgorithm` is configurable by keyword yet. Without it an
# unrecognized keyword surfaces as a `MethodError` on this internal function rather than as a
# complaint about the keyword the caller actually passed.
function reject_algorithm_kwargs(algorithm; kwargs...)
    isempty(kwargs) && return nothing
    names = join(map(k -> "`$k`", collect(keys(kwargs))), ", ")
    return throw(
        ArgumentError(
            "unsupported keyword argument(s) $names for contraction algorithm `$(nameof(typeof(algorithm)))`"
        )
    )
end

function select_contract_algorithm(algorithm, a1, a2; kwargs...)
    return throw(
        ArgumentError(
            "`$algorithm` is not a contraction algorithm; pass a `ContractAlgorithm` as `alg`"
        )
    )
end
function select_contract_algorithm(algorithm::ContractAlgorithm, a1, a2; kwargs...)
    reject_algorithm_kwargs(algorithm; kwargs...)
    return algorithm
end
function select_contract_algorithm(algorithm::DefaultContractAlgorithm, a1, a2; kwargs...)
    return default_contract_algorithm(a1, a2; kwargs...)
end
function default_contract_algorithm(a1, a2; kwargs...)
    algorithm = default_contract_algorithm(typeof(a1), typeof(a2))
    reject_algorithm_kwargs(algorithm; kwargs...)
    return algorithm
end
function default_contract_algorithm(A1::Type{<:AbstractArray}, A2::Type{<:AbstractArray})
    return Matricize(MatricizeStyle(MatricizeStyle(A1), MatricizeStyle(A2)))
end
