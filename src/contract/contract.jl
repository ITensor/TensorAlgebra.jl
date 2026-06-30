# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.
# TODO: Add `scaledcontract(a1, labels1, a2, labels2, α) = α * contract(a1, labels1, a2, labels2)`.

# contract (labels)
function contract(a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...)
    if use_int_labels(labels1) || use_int_labels(labels2)
        # Run the contraction-label bookkeeping on integers, then map the derived labels back.
        int1, int2 = to_int_labels(labels1, labels2)
        int_dest = contract_labels(int1, int2)
        a_dest = contract(int_dest, a1, int1, a2, int2; kwargs...)
        return a_dest, from_int_labels(int_dest, labels1, labels2)
    end
    labels_dest = contract_labels(a1, labels1, a2, labels2)
    return contract(labels_dest, a1, labels1, a2, labels2; kwargs...), labels_dest
end
function contract(
        labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2; kwargs...
    )
    t1 = ntuple(i -> labels1[i], Val(ndims(a1)))
    t2 = ntuple(i -> labels2[i], Val(ndims(a2)))
    contracted1 = map(in(t2), t1)
    # Cross into a `Val(K)` method (a function-barrier on the contracted count) so the
    # bipartitioned permutations and the contraction below them are type-stable.
    return _contract(
        Val(count(contracted1)),
        labels_dest,
        a1,
        t1,
        a2,
        t2,
        contracted1;
        kwargs...
    )
end
function _contract(
        ::Val{K}, labels_dest, a1::AbstractArray, labels1, a2::AbstractArray, labels2,
        contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    return contract(biperm_dest..., a1, biperm1..., a2, biperm2...; kwargs...)
end

# contract (bipartitioned permutations)
function contract(
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...
    )
    Ndest_codomain = Val(length(perm1_codomain))
    Ndest = Val(length(perm1_codomain) + length(perm2_domain))
    perm_dest_codomain, perm_dest_domain =
        bipartition(ntuple(identity, Ndest), Ndest_codomain)
    return contract(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end
function contract(
        perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...
    )
    a_dest = allocate_output(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    return contract!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end

# contract! (labels)
function contract!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2;
        kwargs...
    )
    return contractadd!(
        a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...
    )
end
function contract!(
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain;
        kwargs...
    )
    return contractadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
        true, false; kwargs...
    )
end

# contractadd! (labels)
function contractadd!(
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number;
        kwargs...
    )
    return contractopadd!(
        a_dest, labels_dest, identity, a1, labels1, identity, a2, labels2, α, β; kwargs...
    )
end
# contractadd! (bipartitioned permutations)
function contractadd!(
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        kwargs...
    )
    return contractopadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        identity, a1, perm1_codomain, perm1_domain,
        identity, a2, perm2_codomain, perm2_domain,
        α, β; kwargs...
    )
end

# contractopadd! (labels)
function contractopadd!(
        a_dest::AbstractArray, labels_dest,
        op1, a1::AbstractArray, labels1,
        op2, a2::AbstractArray, labels2,
        α::Number, β::Number;
        kwargs...
    )
    t1 = ntuple(i -> labels1[i], Val(ndims(a1)))
    t2 = ntuple(i -> labels2[i], Val(ndims(a2)))
    contracted1 = map(in(t2), t1)
    # Cross into a `Val(K)` method (a function-barrier on the contracted count) so the
    # bipartitioned permutations and the contraction below them are type-stable.
    return _contractopadd!(
        Val(count(contracted1)), a_dest, labels_dest,
        op1, a1, t1, op2, a2, t2, α, β, contracted1; kwargs...
    )
end
function _contractopadd!(
        ::Val{K}, a_dest::AbstractArray, labels_dest,
        op1, a1::AbstractArray, labels1, op2, a2::AbstractArray, labels2,
        α::Number, β::Number, contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    return contractopadd!(
        a_dest, biperm_dest..., op1, a1, biperm1..., op2, a2, biperm2..., α, β; kwargs...
    )
end
# contractopadd! (bipartitioned permutations, algorithm selection)
function contractopadd!(
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        op1, a1::AbstractArray, perm1_codomain, perm1_domain,
        op2, a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        alg = DefaultContractAlgorithm(), kwargs...
    )
    check_input(
        contract!,
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    algorithm = select_contract_algorithm(alg, a1, a2; kwargs...)
    return contractopadd!(
        algorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α, β
    )
end
# contractopadd! (dispatched on the algorithm, bipartitioned permutations)
# Required interface if not using matricized contraction
function contractopadd!(
        algorithm::ContractAlgorithm,
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        op1, a1::AbstractArray, perm1_codomain, perm1_domain,
        op2, a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    return throw(
        MethodError(
            contractopadd!,
            (
                algorithm,
                a_dest, perm_dest_codomain, perm_dest_domain,
                op1, a1, perm1_codomain, perm1_domain,
                op2, a2, perm2_codomain, perm2_domain,
                α, β,
            )
        )
    )
end
