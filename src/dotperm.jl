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
