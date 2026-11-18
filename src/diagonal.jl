using LinearAlgebra: Diagonal

# Preserve `Diagonal` wherever the result is still diagonal. These methods hook the lowest-level
# primitives, so the wrappers built on them preserve it too.

# Permuting the two axes of a square `Diagonal` (identity or transpose) leaves it
# unchanged, so the lazy permutation is the matrix itself.
permuteddims(a::Diagonal, perm) = a

# Same reasoning for the in-place accumulate: skip the permutation and accumulate
# straight onto the destination.
function bipermutedimsconjadd!(
        dest::Diagonal, isconj::Bool, src::Diagonal,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    check_input(bipermutedimsconjadd!, dest, isconj, src, perm_codomain, perm_domain)
    permutedimsconjadd!(dest.diag, isconj, src.diag, (1,), α, β)
    return dest
end

# The bipermutation of a square `Diagonal` is again a square `Diagonal`. This is decided from
# `src` since the axes alone do not encode that the result is square.
function allocate_output(
        ::typeof(bipermutedimsconj), isconj::Bool, src::Diagonal, perm_codomain, perm_domain
    )
    return Diagonal(similar(src.diag))
end

# The `{1,1}` matricization of a `Diagonal` is `a` itself, which keeps it a `Diagonal`
# downstream.
function isoutputview(
        ::typeof(matricizeconj), isconj::Bool, a::Diagonal, perm_codomain::Tuple{Int},
        perm_domain::Tuple{Int}
    )
    return !isconj && isidentitybiperm(perm_codomain, perm_domain)
end
function matricizeconjview(
        isconj::Bool, a::Diagonal, perm_codomain::Tuple{Int}, perm_domain::Tuple{Int}
    )
    return a
end
# A `{1,1}` unmatricize is the identity, so the result stays `Diagonal`.
function unmatricize(
        m::Diagonal,
        axes_codomain::Tuple{<:AbstractUnitRange}, axes_domain::Tuple{<:AbstractUnitRange}
    )
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return m
end
# Any other split is not representable as a `Diagonal`. `similar(m, axes(m))` densifies while
# keeping `m`'s array backend, where a plain `Array` would force the result onto the CPU.
function unmatricize(m::Diagonal, axes_codomain::Tuple, axes_domain::Tuple)
    return unmatricize(copyto!(similar(m, axes(m)), m), axes_codomain, axes_domain)
end

# Contracting two `Diagonal`s to a `{1,1}` destination gives a `Diagonal`. Other destination
# shapes fall back to the generic dense allocation.
function allocate_output(
        ::typeof(contractpermconjalign),
        perm_dest_codomain::Tuple{Int}, perm_dest_domain::Tuple{Int},
        isconj1::Bool, a1::Diagonal, perm1_codomain, perm1_domain,
        isconj2::Bool, a2::Diagonal, perm2_codomain, perm2_domain
    )
    axes_codomain_dest, _ = output_axes(
        contractpermconjalign,
        perm_dest_codomain, perm_dest_domain,
        isconj1, axes(a1), perm1_codomain, perm1_domain,
        isconj2, axes(a2), perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return Diagonal(zero!(similar(a1.diag, T, (only(axes_codomain_dest),))))
end
