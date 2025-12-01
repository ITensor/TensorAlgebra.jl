# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

abstract type ContractAlgorithm end

ContractAlgorithm(alg::ContractAlgorithm) = alg

struct Matricize{Style} <: ContractAlgorithm
    fusion_style::Style
end
Matricize() = Matricize(ReshapeFusion())

function default_contract_algorithm(A1::Type{<:AbstractArray}, A2::Type{<:AbstractArray})
    style1 = FusionStyle(A1)
    style2 = FusionStyle(A2)
    style1 == style2 || error("Styles must match.")
    return style1
end

# Required interface if not using
# matricized contraction.
function contractadd!(
        alg::ContractAlgorithm,
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    return error("Not implemented")
end

function contract(
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        alg = default_contract_algorithm(typeof(a1), typeof(a2)),
        kwargs...,
    )
    return contract(ContractAlgorithm(alg), a1, labels1, a2, labels2; kwargs...)
end

function contract(
        alg::ContractAlgorithm,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    labels_dest = output_labels(contract, alg, a1, labels1, a2, labels2; kwargs...)
    return contract(alg, labels_dest, a1, labels1, a2, labels2; kwargs...), labels_dest
end

function contract(
        labels_dest,
        a1::AbstractArray,
        labels1,
        a2::AbstractArray,
        labels2;
        alg = default_contract_algorithm(typeof(a1), typeof(a2)),
        kwargs...,
    )
    return contract(ContractAlgorithm(alg), labels_dest, a1, labels1, a2, labels2; kwargs...)
end

function contract!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    return contractadd!(a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
end

function contractadd!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number;
        alg = default_contract_algorithm(typeof(a1), typeof(a2)),
        kwargs...,
    )
    contractadd!(
        ContractAlgorithm(alg), a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...
    )
    return a_dest
end

function contract(
        alg::ContractAlgorithm,
        labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    check_input(contract, a1, labels1, a2, labels2)
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contract(alg, biperm_dest, a1, biperm1, a2, biperm2; kwargs...)
end

function contract!(
        alg::ContractAlgorithm,
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    return contractadd!(
        alg, a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...
    )
end

function contractadd!(
        alg::ContractAlgorithm,
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number;
        kwargs...,
    )
    check_input(contract, a_dest, labels_dest, a1, labels1, a2, labels2)
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contractadd!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...)
end

function contract(
        alg::ContractAlgorithm,
        biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2};
        kwargs...,
    )
    check_input(contract, a1, biperm1, a2, biperm2)
    a_dest = allocate_output(contract, biperm_dest, a1, biperm1, a2, biperm2)
    contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2; kwargs...)
    return a_dest
end
