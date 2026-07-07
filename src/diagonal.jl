using LinearAlgebra: Diagonal

# `Diagonal` participates in the `ReshapeFusion` interface like a dense matrix (it fuses with
# the same row/column reshape order), but its structure is preserved wherever the result of
# an operation is still diagonal. These methods hook the lowest-level primitives, so the
# convenience wrappers built on them (`bipermutedims`, `permutedimsadd!`, `add!`, and the
# matrix functions, which all route through `bipermutedimsopadd!` and `allocate_output`)
# preserve `Diagonal`. Structure is given up (via the generic reshape path) only where the
# result genuinely is not diagonal: vectorizing matricizations and `Diagonal`/dense mixing
# (the latter falls back to Base's dense `similar`, since `contract` allocates from a flat
# axis tuple, not a `BiTuple`).

# Permuting the two axes of a square `Diagonal` (identity or transpose) leaves it
# unchanged, so the lazy permutation is the matrix itself.
permuteddims(a::Diagonal, perm) = a

# Same reasoning for the in-place accumulate: skip the permutation and accumulate
# straight onto the destination.
function bipermutedimsopadd!(
        dest::Diagonal, op, src::Diagonal,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    check_input(bipermutedimsopadd!, dest, op, src, perm_codomain, perm_domain)
    _opadd!(dest.diag, op, src.diag, α, β)
    return dest
end

# The bipermutation of a square `Diagonal` is again a square `Diagonal` of the same size (it
# only swaps or keeps the two axes), so allocate the `permutedimsop`/`bipermutedims` output as
# a `Diagonal`. The squareness comes from `src`, not from the axes: an axis-based
# `similar(::BiTuple)` could not preserve it, since row/column axes alone do not encode that
# the result is square.
function allocate_output(
        ::typeof(permutedimsop),
        op,
        src::Diagonal,
        perm_codomain,
        perm_domain
    )
    T = Base.promote_op(op, eltype(src))
    return Diagonal(similar(src.diag, T))
end

# A `Diagonal` is already a matrix; the `(1 codomain, 1 domain)` matricization is the identity
# reshape, so return it directly (maybe-alias, matching `matricize`'s general contract).
matricize(::ReshapeFusion, a::Diagonal, ::Val{1}) = a
function unmatricize(
        ::ReshapeFusion, m::Diagonal,
        ::Tuple{<:AbstractUnitRange}, ::Tuple{<:AbstractUnitRange}
    )
    return m
end
