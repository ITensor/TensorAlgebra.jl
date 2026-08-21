using LinearAlgebra: Diagonal

# `Diagonal` participates in the `ReshapeMatricize` interface like a dense matrix (it fuses with
# the same row/column reshape order), but its structure is preserved wherever the result of
# an operation is still diagonal. These methods hook the lowest-level primitives, so the
# convenience wrappers built on them (`bipermutedims`, `permutedimsadd!`, `add!`, and the
# matrix functions, which all route through `bipermutedimsopadd!` and `allocate_output`)
# preserve `Diagonal`. Structure is given up only where the result genuinely is not diagonal:
# vectorizing matricizations, a non-`{1,1}` bond-split `unmatricize`, `Diagonal`/dense mixing,
# and `contract` patterns other than the single-contracted-leg matmul (`Diagonal * Diagonal`).

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
matricize(::ReshapeMatricize, a::Diagonal, ::Val{1}) = a
# A `{1,1}` unmatricize (one codomain axis, one domain axis) is the endomorphism identity: the
# result stays `Diagonal`, so return `m` directly. The generic `check_input(unmatricize, ...)`
# validates the axis lengths against `m`'s size.
function unmatricize(
        ::ReshapeMatricize, m::Diagonal,
        axes_codomain::Tuple{<:AbstractUnitRange}, axes_domain::Tuple{<:AbstractUnitRange}
    )
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return m
end
# Any other split is a genuine bond-split (for example `unmatricize(D[4×4], (2, 2), (4,))`) whose
# result is not representable as a `Diagonal`, so densify and reshape like a dense matrix.
# `copyto!(similar(m, axes(m)), m)` densifies while preserving `m`'s array backend (a plain
# `Array` would force the result onto the CPU).
function unmatricize(
        style::ReshapeMatricize, m::Diagonal, axes_codomain::Tuple, axes_domain::Tuple
    )
    return unmatricize(style, copyto!(similar(m, axes(m)), m), axes_codomain, axes_domain)
end

# Contracting two `Diagonal`s over a single leg is the matmul/endomorphism pattern
# `Diagonal * Diagonal = Diagonal` (all transpose variants `[i,j]*[j,k]`, `[i,j]*[k,j]`, ...),
# whose `{1,1}` output stays `Diagonal`, so allocate one. Every other output shape (rank-4 outer
# product, scalar full contraction) is not representable as a `Diagonal` and falls back to the
# generic dense allocation, matching `Diagonal`/dense mixing.
function allocate_contract_output(
        a1::Diagonal, a2::Diagonal, T, axes_codomain::Tuple{Any}, axes_domain::Tuple{Any}
    )
    return Diagonal(zero!(similar(a1.diag, T, (only(axes_codomain),))))
end
