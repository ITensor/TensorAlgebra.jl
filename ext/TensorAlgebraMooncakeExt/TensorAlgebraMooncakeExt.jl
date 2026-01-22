module TensorAlgebraMooncakeExt

using Mooncake: Mooncake, @zero_derivative, DefaultCtx
using TensorAlgebra: AbstractBlockPermutation, ContractAlgorithm, allocate_output,
    biperm, blockedperms, check_input, contract, contract!, contract_labels,
    default_contract_algorithm, select_contract_algorithm

Mooncake.tangent_type(::Type{<:AbstractBlockPermutation}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:ContractAlgorithm}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{
    typeof(allocate_output), typeof(contract), Any, Any, Any, Any, Any,
}
@zero_derivative DefaultCtx Tuple{typeof(biperm), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(blockedperms), typeof(contract), Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(check_input), typeof(contract), Any, Any, Any, Any}
@zero_derivative DefaultCtx Tuple{
    typeof(check_input), typeof(contract!), Any, Any, Any, Any, Any, Any,
}
@zero_derivative DefaultCtx Tuple{typeof(contract_labels), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(contract_labels), Any, Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(default_contract_algorithm), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(select_contract_algorithm), Any, Any, Any}

end
