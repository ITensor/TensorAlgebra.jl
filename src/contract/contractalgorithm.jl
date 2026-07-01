abstract type ContractAlgorithm end
ContractAlgorithm(algorithm::ContractAlgorithm) = algorithm

struct DefaultContractAlgorithm <: ContractAlgorithm end

struct Matricize{LeftStyle, RightStyle, OutputStyle} <: ContractAlgorithm
    left_fusion_style::LeftStyle
    right_fusion_style::RightStyle
    output_fusion_style::OutputStyle
end
Matricize(fusion_style) = Matricize(fusion_style, fusion_style, fusion_style)
Matricize() = Matricize(ReshapeFusion())

function select_contract_algorithm(algorithm, a1, a2)
    return error("Not implemented.")
end
function select_contract_algorithm(algorithm::ContractAlgorithm, a1, a2)
    return algorithm
end
function select_contract_algorithm(algorithm::DefaultContractAlgorithm, a1, a2)
    return default_contract_algorithm(a1, a2)
end
function default_contract_algorithm(a1, a2)
    return default_contract_algorithm(typeof(a1), typeof(a2))
end
function default_contract_algorithm(A1::Type{<:AbstractArray}, A2::Type{<:AbstractArray})
    return Matricize(FusionStyle(FusionStyle(A1), FusionStyle(A2)))
end
