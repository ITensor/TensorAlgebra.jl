abstract type ContractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

struct Matricize{Style} <: ContractAlgorithm
    fusion_style::Style
end
Matricize() = Matricize(ReshapeFusion())

function select_contract_algorithm(algorithm, a1::AbstractArray, a2::AbstractArray)
    return error("Not implemented.")
end
function select_contract_algorithm(
        algorithm::ContractAlgorithm, a1::AbstractArray, a2::AbstractArray
    )
    return algorithm
end
function select_contract_algorithm(
        algorithm::DefaultContractAlgorithm, a1::AbstractArray, a2::AbstractArray
    )
    return default_contract_algorithm(a1, a2)
end
function default_contract_algorithm(a1::AbstractArray, a2::AbstractArray)
    return default_contract_algorithm(typeof(a1), typeof(a2))
end
function default_contract_algorithm(A1::Type{<:AbstractArray}, A2::Type{<:AbstractArray})
    return Matricize(FusionStyle(FusionStyle(A1), FusionStyle(A2)))
end
