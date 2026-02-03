# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.
# TODO: Add `scaledcontract(a1, labels1, a2, labels2, α) = α * contract(a1, labels1, a2, labels2)`.

# contract (labels)
function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
    labels_dest = contract_labels(a1, labels1, a2, labels2)
    return contract(labels_dest, a1, labels1, a2, labels2; kwargs...), labels_dest
end
function contract(
        labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...
    )
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contract(
        blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...;
        kwargs...,
    )
end

# contract (bipartitioned permutations)
function contract(
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...,
    )
    Ndest_codomain = Val(length(perm1_codomain))
    Ndest = Val(length(perm1_codomain) + length(perm2_domain))
    perm_dest_codomain, perm_dest_domain = blocks(trivialbiperm(Ndest_codomain, Ndest))
    return contract(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...,
    )
end
function contract(
        perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...,
    )
    a_dest = allocate_output(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
    )
    return contract!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...,
    )
end

# contract! (labels)
function contract!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...,
    )
    return contractadd!(
        a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...
    )
end
function contract!(
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...,
    )
    return contractadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
        true, false; kwargs...,
    )
end

# contractadd! (labels)
function contractadd!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number;
        kwargs...,
    )
    biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
    return contractadd!(
        a_dest, blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...,
        α, β; kwargs...,
    )
end
function contractadd!(
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        alg = DefaultContractAlgorithm(), kwargs...,
    )
    check_input(
        contract!,
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
    )
    algorithm = select_contract_algorithm(alg, a1, a2; kwargs...)
    return contractadd!(
        algorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
        α, β,
    )
end
# contractadd! (dispatched on the algorithm, bipartitioned permutations)
# Required interface if not using matricized contraction
function contractadd!(
        algorithm::ContractAlgorithm,
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number,
    )
    return throw(
        MethodError(
            contractadd!,
            (
                algorithm,
                a_dest, perm_dest_codomain, perm_dest_domain,
                a1, perm1_codomain, perm1_domain,
                a2, perm2_codomain, perm2_domain,
                α, β,
            )
        )
    )
end

# BlockPermutation versions of contract[add][!]
function contract(
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2};
        kwargs...,
    )
    return contract(a1, blocks(biperm1)..., a2, blocks(biperm2)...; kwargs...)
end
function contract(
        biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2};
        kwargs...,
    )
    return contract(
        blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...;
        kwargs...,
    )
end
function contract!(
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2};
        kwargs...,
    )
    return contract!(
        a_dest, blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...;
        kwargs...,
    )
end
function contractadd!(
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number; kwargs...,
    )
    return contractadd!(
        a_dest, blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...,
        α, β; kwargs...,
    )
end
function contractadd!(
        algorithm::ContractAlgorithm,
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    return contractadd!(
        algorithm,
        a_dest, blocks(biperm_dest)...,
        a1, blocks(biperm1)...,
        a2, blocks(biperm2)...,
        α, β,
    )
end
