# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

# Required interface if not using matricized contraction.
function contractadd!(
        alg::ContractAlgorithm,
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    return error("Not implemented")
end

# contract
function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
    labels_dest = contract_labels(a1, labels1, a2, labels2)
    return contract(labels_dest, a1, labels1, a2, labels2; kwargs...), labels_dest
end
function contract(
        labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...
    )
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contract(biperm_dest, a1, biperm1, a2, biperm2; kwargs...)
end
function contract(
        biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2};
        kwargs...,
    )
    a_dest = allocate_output(contract, biperm_dest, a1, biperm1, a2, biperm2)
    return contract!(a_dest, biperm_dest, a1, biperm1, a2, biperm2; kwargs...)
end

# contract!
function contract!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    return contractadd!(a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...)
end

# contractadd!
function contractadd!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number;
        kwargs...,
    )
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contractadd!(a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...)
end
function contractadd!(
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number;
        alg = DefaultContractAlgorithm(), kwargs...,
    )
    check_input(contract!, a_dest, biperm_dest, a1, biperm1, a2, biperm2)
    alg′ = select_contract_algorithm(alg, a1, a2; kwargs...)
    return contractadd!(alg′, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β)
end
