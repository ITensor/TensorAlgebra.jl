using LinearAlgebra: LinearAlgebra

"""
    dotperm(a, b, perm)
    dotperm(a, b, perm_codomain, perm_domain)

Compute the inner product of `a` with `b` after aligning the dimensions of `b`.
The first argument is conjugated, following `LinearAlgebra.dot`.

The groups `perm_codomain` and `perm_domain` must partition the dimensions of `b`,
with `length(perm_codomain) == ndims_codomain(a)`. Both operands are matricized
with the codomain/domain split of `a`, then paired using `LinearAlgebra.dot`
on their native matrix representations. The three-argument form uses an empty domain.

See also [`matricize`](@ref) and [`ndims_codomain`](@ref).
"""
function dotperm(a, b, perm)
    return dotperm(a, b, perm, ())
end

function dotperm(a, b, perm_codomain, perm_domain)
    check_biperm(b, perm_codomain, perm_domain)
    length(perm_codomain) == ndims_codomain(a) || throw(
        ArgumentError(
            "The codomain permutation must match the codomain dimension count of the first operand"
        )
    )
    return LinearAlgebra.dot(
        matricize(a, Val(ndims_codomain(a))), matricize(b, perm_codomain, perm_domain)
    )
end

"""
    TensorAlgebra.dot(a, labels_a, b, labels_b)

Compute the inner product of `a` and `b`, aligning the dimensions of `b` by their labels.
Each operand must have one unique label per dimension, and both label sets must match.
The first argument is conjugated, following `LinearAlgebra.dot`.

This is `TensorAlgebra`'s own function, distinct from `LinearAlgebra.dot`. It preserves
`a`'s codomain/domain split and delegates the aligned pairing to [`dotperm`](@ref).

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
    labels_a, labels_b = Tuple(labels_a), Tuple(labels_b)
    length(labels_a) == ndims(a) && length(labels_b) == ndims(b) ||
        throw(ArgumentError("Each operand must have one label per dimension"))
    allunique(labels_a) && allunique(labels_b) && issetequal(labels_a, labels_b) ||
        throw(ArgumentError("Inner product labels must be unique and matching"))
    perm = map(label -> something(findfirst(isequal(label), labels_b)), labels_a)
    return dotperm(a, b, bipartition(perm, Val(ndims_codomain(a)))...)
end
