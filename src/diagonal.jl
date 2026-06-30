using LinearAlgebra: Diagonal

# `Diagonal` participates in the `ReshapeFusion` interface like a dense matrix (it fuses with
# the same row/column reshape order), but its structure is preserved wherever the result of
# an operation is still diagonal. These methods hook the lowest-level primitives, so the
# convenience wrappers built on them (`bipermutedims`, `permutedimsadd!`, `add!`, and the
# matrix functions, which all route through `bipermutedimsopadd!` and `similar`) preserve
# `Diagonal` for free. Structure is given up (via the generic reshape path) only where the
# result genuinely is not diagonal: vectorizing matricizations and `Diagonal`/dense mixing.

# Permuting the two axes of a square `Diagonal` (identity or transpose) leaves the stored
# diagonal unchanged, so accumulate straight onto it.
function bipermutedimsopadd!(
        dest::Diagonal, op, src::Diagonal,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    check_input(bipermutedimsopadd!, dest, op, src, perm_codomain, perm_domain)
    _opadd!(dest.diag, op, src.diag, α, β)
    return dest
end

# Bipartitioned allocation of a `Diagonal` stays a `Diagonal` (the permuted output is still
# 2-dimensional and square), so `permutedimsop`/`bipermutedims` of a `Diagonal` allocate a
# `Diagonal`. `BiTuple` is owned by TensorAlgebra, so this is not piracy.
function Base.similar(a::Diagonal, T::Type, axes::BiTuple)
    return Diagonal(similar(a.diag, T, length(first(Tuple(axes)))))
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
