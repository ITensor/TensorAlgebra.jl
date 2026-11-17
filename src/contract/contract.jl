# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.
# TODO: Add `scaledcontract(a1, labels1, a2, labels2, α) = α * contract(a1, labels1, a2, labels2)`.

# contract (labels)
"""
    contract(a1, labels1, a2, labels2; alg = nothing) -> a_dest, labels_dest

Contract the arrays over the labels they share, returning the result along with the labels of its
dimensions. A label appearing on two operands is summed over, one appearing on a single operand
survives, and `labels_dest` reports the surviving labels in the order the result carries them.

```jldoctest
julia> using TensorAlgebra: contract

julia> a, b = randn(2, 3), randn(3, 4);

julia> ab, labels = contract(a, (:i, :j), b, (:j, :k));

julia> labels
2-element Vector{Symbol}:
 :i
 :k

julia> ab ≈ a * b
true
```

See also [`contractalign`](@ref) to name the output dimensions and get back just the array.
"""
function contract(a1, labels1, a2, labels2; kwargs...)
    # A no-op unless the label type opts in through `label_type`.
    l1, l2 = encode_contraction_labels(labels1, labels2)
    l_dest = contract_labels(l1, l2)
    a_dest = contractalign(l_dest, a1, l1, a2, l2; kwargs...)
    return a_dest, decode_contraction_labels(l_dest, labels1, labels2)
end

"""
    contractalign(labels_dest, a1, labels1, a2, labels2; alg = nothing) -> a_dest

Contract the input arrays over the shared labels, aligning the output array according to the
specified destination labels `labels_dest`. `labels_dest` must match the uncontracted labels,
i.e. `issetequal(labels_dest, symdiff(labels1, labels2))` must be `true`.

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
    return contractconj(labels_dest, false, a1, labels1, false, a2, labels2; kwargs...)
end

# The `op` forms take each operand's operation as `identity` or `conj` and forward to the `conj`
# forms, which take a conjugation flag and are the ones to overload.

# contractop (labels, with operations on the operands)
"""
    contractop(labels_dest, op1, a1, labels1, op2, a2, labels2; alg = nothing) -> a_dest

Contract `op1(a1)` with `op2(a2)` over their shared labels, with the dimensions of the result
ordered by `labels_dest`. Each `op` is `identity` or `conj`.

See also [`contractconj`](@ref).
"""
function contractop(labels_dest, op1, a1, labels1, op2, a2, labels2; kwargs...)
    return contractconj(
        labels_dest, _isconj(op1), a1, labels1, _isconj(op2), a2, labels2; kwargs...
    )
end
"""
    contractconj(labels_dest, isconj1::Bool, a1, labels1, isconj2::Bool, a2, labels2;
                 alg = nothing) -> a_dest

Contract `a1` with `a2` over their shared labels, with the dimensions of the result ordered
by `labels_dest`, conjugating each operand whose flag is `true`.

See also [`contractop`](@ref).
"""
function contractconj(
        labels_dest, isconj1::Bool, a1, labels1, isconj2::Bool, a2, labels2; kwargs...
    )
    isconj1, a1 = _unwrap_conj(isconj1, a1)
    isconj2, a2 = _unwrap_conj(isconj2, a2)
    t1 = ntuple(i -> labels1[i], Val(ndims(a1)))
    t2 = ntuple(i -> labels2[i], Val(ndims(a2)))
    contracted1 = map(in(t2), t1)
    # Computed from the dimensions, so it is a compile-time constant when `labels_dest` is a tuple.
    K = (ndims(a1) + ndims(a2) - length(labels_dest)) ÷ 2
    count(contracted1) == K || throw(ArgumentError("Invalid contraction labels"))
    # Otherwise `K` is only known at run time, and inferring `_contractconj` for an unknown `Val`
    # would analyze the whole contraction with abstract argument types for nothing.
    f = labels_dest isa Tuple ? _contractconj : Base.inferencebarrier(_contractconj)
    return f(
        Val(K), labels_dest, isconj1, a1, t1, isconj2, a2, t2, contracted1; kwargs...
    )
end
function _contractconj(
        ::Val{K}, labels_dest, isconj1::Bool, a1, labels1, isconj2::Bool, a2, labels2,
        contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    a_dest = allocate_output(
        contractpermconj, biperm_dest..., isconj1, a1, biperm1..., isconj2, a2,
        biperm2...
    )
    return contractpermconjadd!(
        a_dest, biperm_dest..., isconj1, a1, biperm1..., isconj2, a2, biperm2..., true,
        false;
        kwargs...
    )
end

# contractperm (bipartitioned permutations)
function contractperm(
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    Ndest_codomain = Val(length(perm1_codomain))
    Ndest = Val(length(perm1_codomain) + length(perm2_domain))
    perm_dest_codomain, perm_dest_domain =
        bipartition(ntuple(identity, Ndest), Ndest_codomain)
    return contractpermalign(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end
function contractpermalign(
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    return contractpermconj(
        perm_dest_codomain, perm_dest_domain,
        false, a1, perm1_codomain, perm1_domain,
        false, a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end
"""
    contractpermop(perm_dest_codomain, perm_dest_domain,
                   op1, a1, perm1_codomain, perm1_domain,
                   op2, a2, perm2_codomain, perm2_domain; alg = nothing) -> a_dest

Contract `op1(a1)` with `op2(a2)` over the dimensions `perm1_domain` of `a1` and
`perm2_codomain` of `a2`, with the result's dimensions in the order given by the destination
permutations. Each `op` is `identity` or `conj`.

See also [`contractpermconj`](@ref).
"""
function contractpermop(
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    return contractpermconj(
        perm_dest_codomain, perm_dest_domain,
        _isconj(op1), a1, perm1_codomain, perm1_domain,
        _isconj(op2), a2, perm2_codomain, perm2_domain;
        kwargs...
    )
end
"""
    contractpermconj(perm_dest_codomain, perm_dest_domain,
                     isconj1::Bool, a1, perm1_codomain, perm1_domain,
                     isconj2::Bool, a2, perm2_codomain, perm2_domain; alg = nothing) -> a_dest

Contract `a1` with `a2` over the dimensions `perm1_domain` of `a1` and `perm2_codomain` of
`a2`, conjugating each operand whose flag is `true`, with the result's dimensions in the order
given by the destination permutations.

See also [`contractpermop`](@ref).
"""
function contractpermconj(
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain;
        kwargs...
    )
    isconj1, a1 = _unwrap_conj(isconj1, a1)
    isconj2, a2 = _unwrap_conj(isconj2, a2)
    a_dest = allocate_output(
        contractpermconj,
        perm_dest_codomain, perm_dest_domain,
        isconj1, a1, perm1_codomain, perm1_domain,
        isconj2, a2, perm2_codomain, perm2_domain
    )
    return contractpermconjadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1, a1, perm1_codomain, perm1_domain,
        isconj2, a2, perm2_codomain, perm2_domain,
        true, false; kwargs...
    )
end

# contract! (labels)
"""
    contract!(a_dest, labels_dest, a1, labels1, a2, labels2; alg = nothing) -> a_dest

Overwrite `a_dest`, with dimensions labeled by `labels_dest`, with the contraction of `a1` and
`a2` over their shared labels.

See also [`contract`](@ref) and [`contractadd!`](@ref).
"""
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
"""
    contractadd!(a_dest, labels_dest, a1, labels1, a2, labels2, α, β; alg = nothing) -> a_dest

`a_dest = β * a_dest + α * c`, where `c` is the contraction of `a1` and `a2` over their shared
labels and `labels_dest` labels the dimensions of `a_dest`.

See also [`contract!`](@ref).
"""
function contractadd!(
        a_dest, labels_dest,
        a1, labels1,
        a2, labels2,
        α::Number, β::Number;
        kwargs...
    )
    return contractconjadd!(
        a_dest, labels_dest, false, a1, labels1, false, a2, labels2, α, β; kwargs...
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
    return contractpermconjadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        false, a1, perm1_codomain, perm1_domain,
        false, a2, perm2_codomain, perm2_domain,
        α, β; kwargs...
    )
end

# contractopadd! (labels)
"""
    contractopadd!(a_dest, labels_dest, op1, a1, labels1, op2, a2, labels2, α, β;
                   alg = nothing) -> a_dest

`a_dest = β * a_dest + α * c`, where `c` is the contraction of `op1(a1)` and `op2(a2)` over
their shared labels. Each `op` is `identity` or `conj`.

See also [`contractconjadd!`](@ref).
"""
function contractopadd!(
        a_dest, labels_dest,
        op1, a1, labels1,
        op2, a2, labels2,
        α::Number, β::Number;
        kwargs...
    )
    return contractconjadd!(
        a_dest, labels_dest, _isconj(op1), a1, labels1, _isconj(op2), a2, labels2,
        α, β; kwargs...
    )
end
"""
    contractconjadd!(a_dest, labels_dest, isconj1::Bool, a1, labels1,
                     isconj2::Bool, a2, labels2, α, β; alg = nothing) -> a_dest

`a_dest = β * a_dest + α * c`, where `c` is the contraction of `a1` and `a2` over their shared
labels, conjugating each operand whose flag is `true`.

See also [`contractopadd!`](@ref).
"""
function contractconjadd!(
        a_dest, labels_dest,
        isconj1::Bool, a1, labels1,
        isconj2::Bool, a2, labels2,
        α::Number, β::Number;
        kwargs...
    )
    isconj1, a1 = _unwrap_conj(isconj1, a1)
    isconj2, a2 = _unwrap_conj(isconj2, a2)
    t1 = ntuple(i -> labels1[i], Val(ndims(a1)))
    t2 = ntuple(i -> labels2[i], Val(ndims(a2)))
    contracted1 = map(in(t2), t1)
    # Computed from the dimensions, so it is a compile-time constant.
    K = (ndims(a1) + ndims(a2) - ndims(a_dest)) ÷ 2
    count(contracted1) == K || throw(ArgumentError("Invalid contraction labels"))
    return _contractconjadd!(
        Val(K), a_dest, labels_dest,
        isconj1, a1, t1, isconj2, a2, t2, α, β, contracted1; kwargs...
    )
end
function _contractconjadd!(
        ::Val{K}, a_dest, labels_dest,
        isconj1::Bool, a1, labels1, isconj2::Bool, a2, labels2,
        α::Number, β::Number, contracted1; kwargs...
    ) where {K}
    biperm_dest, biperm1, biperm2 =
        biperms(contract, Val(K), labels_dest, labels1, labels2, contracted1)
    return contractpermconjadd!(
        a_dest, biperm_dest..., isconj1, a1, biperm1..., isconj2, a2, biperm2..., α, β;
        kwargs...
    )
end
# contractpermopadd! (bipartitioned permutations, algorithm selection)
"""
    contractpermopadd!(a_dest, perm_dest_codomain, perm_dest_domain,
                       op1, a1, perm1_codomain, perm1_domain,
                       op2, a2, perm2_codomain, perm2_domain, α, β; alg = nothing) -> a_dest

`a_dest = β * a_dest + α * c`, where `c` is the contraction of `op1(a1)` and `op2(a2)` as in
[`contractpermop`](@ref).

See also [`contractpermconjadd!`](@ref).
"""
function contractpermopadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        kwargs...
    )
    return contractpermconjadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        _isconj(op1), a1, perm1_codomain, perm1_domain,
        _isconj(op2), a2, perm2_codomain, perm2_domain,
        α, β; kwargs...
    )
end
"""
    contractpermconjadd!(a_dest, perm_dest_codomain, perm_dest_domain,
                         isconj1::Bool, a1, perm1_codomain, perm1_domain,
                         isconj2::Bool, a2, perm2_codomain, perm2_domain, α, β;
                         alg = nothing) -> a_dest
    contractpermconjadd!(alg::ContractAlgorithm, a_dest, perm_dest_codomain, perm_dest_domain,
                         isconj1::Bool, a1, perm1_codomain, perm1_domain,
                         isconj2::Bool, a2, perm2_codomain, perm2_domain, α, β) -> a_dest

`a_dest = β * a_dest + α * c`, where `c` is the contraction of `a1` and `a2` as in
[`contractpermconj`](@ref). Array types and contraction backends overload the second form,
which takes the [`ContractAlgorithm`](@ref) first.

See also [`contractpermopadd!`](@ref).
"""
function contractpermconjadd!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number;
        alg = nothing
    )
    isconj1, a1 = _unwrap_conj(isconj1, a1)
    isconj2, a2 = _unwrap_conj(isconj2, a2)
    algorithm = select_algorithm(contract!, alg, (a_dest, a1, a2))
    return contractpermconjadd!(
        algorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1, a1, perm1_codomain, perm1_domain,
        isconj2, a2, perm2_codomain, perm2_domain,
        α, β
    )
end
# contractpermconjadd! (dispatched on the algorithm, bipartitioned permutations)
# Required interface of a contraction backend other than `MatricizeContract`.
function contractpermconjadd!(
        algorithm::ContractAlgorithm,
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    return throw(
        MethodError(
            contractpermconjadd!,
            (
                algorithm,
                a_dest, perm_dest_codomain, perm_dest_domain,
                isconj1, a1, perm1_codomain, perm1_domain,
                isconj2, a2, perm2_codomain, perm2_domain,
                α, β,
            )
        )
    )
end

# Keyed on `contract!` rather than `contract` so a backend can choose on the destination as well
# as on the operands.
function default_algorithm(
        ::typeof(contract!), ::Type{<:Tuple{AbstractArray, AbstractArray, AbstractArray}}
    )
    return MatricizeContract()
end
function select_algorithm(::typeof(contract!), ::DefaultContractAlgorithm, args::Tuple)
    return default_algorithm(contract!, args)
end
