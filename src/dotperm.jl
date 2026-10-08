using LinearAlgebra: LinearAlgebra

"""
    dotperm(a, b, perm)

Compute the inner product of `a` and `b` after applying the full dimension permutation
`perm` to `b`, using the codomain/domain split of `a`.
The permuted axes must match: `axes(a, i) == axes(b, perm[i])`.

# Examples

```jldoctest
julia> import TensorAlgebra

julia> TensorAlgebra.dotperm([1 2; 3 4], [5 6; 7 8], (2, 1))
69
```

See also [`dot`](@ref).
"""
function dotperm(a, b, perm)
    check_input(dotperm, a, b, perm)
    perm = NTuple{ndims(a)}(perm)
    perm_codomain, perm_domain = bipartition(perm, Val(ndims_codomain(a)))
    return LinearAlgebra.dot(
        matricize(a, Val(ndims_codomain(a))), matricize(b, perm_codomain, perm_domain)
    )
end

function check_input(::typeof(dotperm), a, b, perm)
    length(perm) == ndims(a) ||
        throw(
        ArgumentError(
            "Permutation length must match the first operand's dimension count"
        )
    )
    check_biperm(b, perm, ())
    for i in 1:ndims(a)
        axes(a, i) == axes(b, perm[i]) || throw(
            DimensionMismatch(
                "Inner product axes do not match: axes(a, $i) = $(axes(a, i)), axes(b, $(perm[i])) = $(axes(b, perm[i]))"
            )
        )
    end
    return nothing
end

"""
    TensorAlgebra.dot(a, labels_a, b, labels_b)

Compute the inner product of `a` and `b` after permuting the dimensions of `b` to match
the labels of `a`. Each operand must have one unique label per dimension, and both label
sets must match.

# Examples

```jldoctest
julia> import TensorAlgebra

julia> a = [1 2; 3 4];
       b = [5 6; 7 8];

julia> TensorAlgebra.dot(a, (:i, :j), b, (:j, :i))
69
```

See also [`dotperm`](@ref) and [`matricize`](@ref).
"""
function dot(a, labels_a, b, labels_b)
    check_input(dot, a, labels_a, b, labels_b)
    labels_a_tuple = NTuple{ndims(a)}(labels_a)
    labels_b_tuple = NTuple{ndims(b)}(labels_b)
    perm = tuple_indexin(labels_a_tuple, labels_b_tuple)
    return dotperm(a, b, perm)
end

function check_input(::typeof(dot), a, labels_a, b, labels_b)
    length(labels_a) == ndims(a) && length(labels_b) == ndims(b) ||
        throw(ArgumentError("Each operand must have one label per dimension"))
    allunique(labels_a) && allunique(labels_b) && issetequal(labels_a, labels_b) ||
        throw(ArgumentError("Inner product labels must be unique and matching"))
    return nothing
end
