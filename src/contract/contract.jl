# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.
# TODO: Add `scaledcontract(a1, labels1, a2, labels2, α) = α * contract(a1, labels1, a2, labels2)`.

# contract (labels)
"""
    contract(a1, labels1, a2, labels2, ...; alg = nothing) -> a_dest, labels_dest

Contract the arrays over the labels they share, returning the result along with the labels of its
dimensions. A label appearing on two operands is summed over, one appearing on a single operand
survives, and `labels_dest` reports the surviving labels in the order the result carries them.

Operands past the second are contracted one pair at a time from left to right, so the call
expresses the contraction order rather than requesting an optimized one.

```jldoctest
julia> using TensorAlgebra: contract

julia> a, b = randn(2, 3), randn(3, 4);

julia> ab, labels = contract(a, (:i, :j), b, (:j, :k));

julia> labels
(:i, :k)

julia> ab ≈ a * b
true
```

See also [`contractalign`](@ref) to name the output dimensions and get back just the array.
"""
function contract(a1, labels1, a2, labels2; kwargs...)
    # Optionally convert the labels to a representation cheaper to run the bookkeeping on (see
    # `label_type`). `encode_contraction_labels`/`decode_contraction_labels` are no-ops unless the label type opts in.
    l1, l2 = encode_contraction_labels(labels1, labels2)
    l_dest = contract_labels(l1, l2)
    a_dest = contractalign(l_dest, a1, l1, a2, l2; kwargs...)
    return a_dest, decode_contraction_labels(l_dest, labels1, labels2)
end
function contract(a1, labels1, a2, labels2, a3, labels3, rest...; kwargs...)
    check_alternating_labels(contract, rest)
    a12, labels12 = contract(a1, labels1, a2, labels2; kwargs...)
    return contract(a12, labels12, a3, labels3, rest...; kwargs...)
end

"""
    contractalign(labels_dest, a1, labels1, a2, labels2, ...; alg = nothing) -> a_dest

Contract the arrays over the labels they share into a result whose dimensions carry
`labels_dest`, which must be the surviving labels in some order.

This is [`contract`](@ref) with the output specified, so it returns the array on its own. The
name matches `ITensorBase.align`: arrange the result's dimensions to match the labels given.

```jldoctest
julia> using TensorAlgebra: contractalign

julia> a, b = randn(2, 3), randn(3, 4);

julia> contractalign((:k, :i), a, (:i, :j), b, (:j, :k)) ≈ permutedims(a * b, (2, 1))
true
```
"""
function contractalign(
        labels_dest, a1, labels1, a2, labels2; kwargs...
    )
    t1 = ntuple(i -> labels1[i], Val(ndims(a1)))
    t2 = ntuple(i -> labels2[i], Val(ndims(a2)))
    contracted1 = map(in(t2), t1)
    # Cross into a `Val(K)` method (a function-barrier on the contracted count) so the
    # bipartitioned permutations and the contraction below them are type-stable.
    return _contractalign(
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
# Only the last pair lands on the requested output labels; the ones before it infer their own.
function contractalign(
        labels_dest, a1, labels1, a2, labels2, a3, labels3, rest...; kwargs...
    )
    check_alternating_labels(contractalign, rest)
    a12, labels12 = contract(a1, labels1, a2, labels2; kwargs...)
    return contractalign(labels_dest, a12, labels12, a3, labels3, rest...; kwargs...)
end
function _contractalign(
        ::Val{K}, labels_dest, a1, labels1, a2, labels2,
        contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    return contractperm(biperm_dest..., a1, biperm1..., a2, biperm2...; kwargs...)
end

# The variadic forms take arrays and labels in alternating positions, so a trailing group with an
# odd length is a miscount at the call site rather than something to diagnose further down.
function check_alternating_labels(f, rest::Tuple)
    iseven(length(rest)) || throw(
        ArgumentError(
            "`$f` takes each array followed by its labels, so the trailing arguments must come in pairs"
        )
    )
    return nothing
end

# contractperm (bipartitioned permutations)
# `perm` marks the whole biperm ladder: every rung has a labels-form sibling under the plain name,
# and once `contract` is variadic over operands the two can no longer be told apart by arity.
function contractperm(
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    Ndest_codomain = Val(length(perm1_codomain))
    Ndest = Val(length(perm1_codomain) + length(perm2_domain))
    perm_dest_codomain, perm_dest_domain =
        bipartition(ntuple(identity, Ndest), Ndest_codomain)
    return contractperm(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end
function contractperm(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    a_dest = allocate_output(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    return contractperm!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end

# contract! (labels)
function contract!(
        a_dest, labels_dest,
        a1, labels1,
        a2, labels2;
        kwargs...
    )
    return contractadd!(
        a_dest, labels_dest, a1, labels1, a2, labels2, true, false; kwargs...
    )
end
function contractperm!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    return contractpermadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
        true, false; kwargs...
    )
end

# contractadd! (labels)
function contractadd!(
        a_dest, labels_dest,
        a1, labels1,
        a2, labels2,
        α::Number, β::Number;
        kwargs...
    )
    return contractopadd!(
        a_dest, labels_dest, identity, a1, labels1, identity, a2, labels2, α, β; kwargs...
    )
end
# contractpermadd! (bipartitioned permutations)
function contractpermadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        kwargs...
    )
    return contractpermopadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        identity, a1, perm1_codomain, perm1_domain,
        identity, a2, perm2_codomain, perm2_domain,
        α, β; kwargs...
    )
end

# contractopadd! (labels)
function contractopadd!(
        a_dest, labels_dest,
        op1, a1, labels1,
        op2, a2, labels2,
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
        ::Val{K}, a_dest, labels_dest,
        op1, a1, labels1, op2, a2, labels2,
        α::Number, β::Number, contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    return contractpermopadd!(
        a_dest, biperm_dest..., op1, a1, biperm1..., op2, a2, biperm2..., α, β; kwargs...
    )
end
# contractpermopadd! (bipartitioned permutations, algorithm selection)
function contractpermopadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        alg = nothing, kwargs...
    )
    check_input(
        contract!,
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    algorithm = select_algorithm(contract, a1, a2; alg, kwargs...)
    return contractpermopadd!(
        algorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α, β
    )
end
# contractpermopadd! (dispatched on the algorithm, bipartitioned permutations)
# Required interface if not using matricized contraction
function contractpermopadd!(
        algorithm::ContractAlgorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    return throw(
        MethodError(
            contractpermopadd!,
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

# Bridges from the operation-generic algorithm layer in `algorithm.jl` down to the contraction
# resolvers. They live here rather than beside those resolvers because dispatching on
# `::typeof(contract)` needs `contract` to exist, and `contractalgorithm.jl` is included first for
# the algorithm types this file's signatures use.
function default_algorithm(::typeof(contract), A1::Type, A2::Type; kwargs...)
    algorithm = default_contract_algorithm(A1, A2)
    reject_algorithm_kwargs(algorithm; kwargs...)
    return algorithm
end
function select_algorithm_specified(
        ::typeof(contract), alg::ContractAlgorithm, a1, a2; kwargs...
    )
    return select_contract_algorithm(alg, a1, a2; kwargs...)
end
