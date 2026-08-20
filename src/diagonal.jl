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
        codomain_axes::Tuple{<:AbstractUnitRange}, domain_axes::Tuple{<:AbstractUnitRange}
    )
    check_input(unmatricize, m, codomain_axes, domain_axes)
    return m
end
# Any other split is a genuine bond-split (for example `unmatricize(D[4×4], (2, 2), (4,))`) whose
# result is not representable as a `Diagonal`, so densify and reshape like a dense matrix.
# `copyto!(similar(m, axes(m)), m)` densifies while preserving `m`'s array backend (a plain
# `Array` would force the result onto the CPU).
function unmatricize(
        style::ReshapeMatricize, m::Diagonal, codomain_axes::Tuple, domain_axes::Tuple
    )
    return unmatricize(style, copyto!(similar(m, axes(m)), m), codomain_axes, domain_axes)
end

# Contracting two `Diagonal`s over a single leg is the matmul/endomorphism pattern
# `Diagonal * Diagonal = Diagonal` (all transpose variants `[i,j]*[j,k]`, `[i,j]*[k,j]`, ...),
# whose result is again a `Diagonal`, so allocate one. Every other pattern is not representable
# as a `Diagonal`: an outer product (no contracted leg) is rank 4, a full contraction (both
# legs) is a scalar, and a non-`{1,1}` output split vectorizes the result; those fall back to
# the generic dense allocation, matching `Diagonal`/dense mixing.
function allocate_output(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1::Diagonal, perm1_codomain, perm1_domain,
        a2::Diagonal, perm2_codomain, perm2_domain
    )
    check_input(
        contract, a1, perm1_codomain, perm1_domain, a2, perm2_codomain, perm2_domain
    )
    codomain_axes_dest, domain_axes_dest = output_axes(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return diagonal_contract_output(
        a1, T, codomain_axes_dest, domain_axes_dest,
        Val(length(perm_dest_codomain)), Val(length(perm_dest_domain))
    )
end

# A `{1,1}` output (one codomain leg, one domain leg) stays `Diagonal`: for two diagonals that is the
# single-contracted-leg matmul pattern. Every other output shape (rank-4 outer product, scalar full
# contraction) densifies through `similar_map`. `similar_map` dualizes the domain axes itself, and
# for a dense `Diagonal` those axes are `Base.OneTo`, where dualizing is a no-op anyway.
function diagonal_contract_output(
        a1::Diagonal, T, codomain_axes, domain_axes, ndims_codomain::Val{1},
        ndims_domain::Val{1}
    )
    return Diagonal(zero!(similar(a1.diag, T, length(only(codomain_axes)))))
end
function diagonal_contract_output(
        a1::Diagonal, T, codomain_axes, domain_axes, ndims_codomain::Val, ndims_domain::Val
    )
    return zero!(similar_map(a1, T, codomain_axes, domain_axes))
end
