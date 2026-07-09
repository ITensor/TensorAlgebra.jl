using EllipsisNotation: Ellipsis
using LinearAlgebra: Diagonal

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

FusionStyle(x) = FusionStyle(typeof(x))
FusionStyle(T::Type) = throw(MethodError(FusionStyle, (T,)))
FusionStyle(style1::Style, style2::Style) where {Style <: FusionStyle} = Style()
FusionStyle(style1::FusionStyle, style2::FusionStyle) = ReshapeFusion()

# =======================================  misc  ========================================

"""
    TensorAlgebra.trivialrange(R::Type{<:AbstractUnitRange}[, n::Integer])
    TensorAlgebra.trivialrange(r::AbstractUnitRange[, n::Integer])

Return the identity range for fusing ranges of type `R`: a one-dimensional range
`t` for which fusing `t` with any other range of the same family leaves that range
unchanged. Defaults to `Base.OneTo(1)`. Downstream packages overload the type-level
methods to return their own identity (for example, a charge-0 one-dimensional sector
for a graded range).

With a length `n`, return the `n`-dimensional analogue: `n` copies of the identity
range stacked into one range (for a graded range, a charge-0 sector of dimension `n`).
"""
trivialrange(r::AbstractUnitRange) = trivialrange(typeof(r))
trivialrange(::Type{<:AbstractUnitRange}) = Base.OneTo(1)
trivialrange(r::AbstractUnitRange, n::Integer) = trivialrange(typeof(r), n)
trivialrange(::Type{<:AbstractUnitRange}, n::Integer) = Base.OneTo(n)

"""
    permutedimsop(op, src, perm_codomain, perm_domain)

Non-mutating version of `bipermutedimsopadd!`: returns
`op.(permutedims(src, (perm_codomain..., perm_domain...)))`.
"""
function permutedimsop(op, src, perm_codomain, perm_domain)
    dest = allocate_output(permutedimsop, op, src, perm_codomain, perm_domain)
    return bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, true, false)
end

# The output holds `op.(src)` permuted, so `op` applies to the axes too: `conj` dualizes a
# graded axis (a no-op on a dense axis), `identity` leaves it unchanged, keeping axes and
# data in sync.
function allocate_output(::typeof(permutedimsop), op, src, perm_co, perm_do)
    T = Base.promote_op(op, eltype(src))
    axes_co = map(i -> op(axes(src, i)), perm_co)
    axes_do = map(i -> op(axes(src, i)), perm_do)
    # `axes_do` are in the stored/dualized convention (from `axes(src)`), so un-dualize them
    # into `similar_map`'s codomain-facing convention.
    return similar_map(src, T, axes_co, conj.(axes_do))
end

function bipermutedims(a, perm1, perm2)
    return permutedimsop(identity, a, perm1, perm2)
end
function bipermutedims!(a_dest, a_src, perm1, perm2)
    return bipermutedimsopadd!(a_dest, identity, a_src, perm1, perm2, true, false)
end
function bipermutedims(a, biperm::BiTuple)
    return bipermutedims(a, biperm.t1, biperm.t2)
end
function bipermutedims!(
        a_dest, a_src, biperm::BiTuple
    )
    return bipermutedims!(a_dest, a_src, biperm.t1, biperm.t2)
end

# =====================================  matricize  ========================================
# TBD settle copy/not copy convention
# matrix factorizations assume copy
# maybe: copy=false kwarg

# This is the primary function that should be overloaded for new fusion styles.
# This assumes the permutation was already performed.
function matricize(
        style::FusionStyle, a, ndims_codomain::Val
    )
    return throw(MethodError(matricize, (style, a, ndims_codomain)))
end
function matricize(a, ndims_codomain::Val)
    return matricize(FusionStyle(a), a, ndims_codomain)
end

function matricizeperm(
        a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    return matricizeperm(FusionStyle(a), a, perm_codomain, perm_domain)
end
# Thin wrapper around `matricizeopperm` with identity op — the actual matricization logic
# (and the fusion-style overload point for folding ops into matricization) lives in
# `matricizeopperm`.
function matricizeperm(
        style::FusionStyle, a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    return matricizeopperm(style, identity, a, perm_codomain, perm_domain)
end

# Process inputs such as `EllipsisNotation.Ellipsis`.
function to_permblocks(a, permblocks::NTuple{2, Tuple{Vararg{Int}}})
    isperm((permblocks[1]..., permblocks[2]...)) ||
        throw(ArgumentError("Invalid bipermutation"))
    return permblocks
end
# Like `setcomplement` is like `setdiff` but assumes t2 ⊆ t1.
function tuplesetcomplement(t1::NTuple{N1}, t2::NTuple{N2}) where {N1, N2}
    t2 ⊆ t1 || throw(ArgumentError("t2 must be a subset of t1"))
    return NTuple{N1 - N2}(setdiff(t1, t2))
end
function to_permblocks(
        a, permblocks::Tuple{Tuple{Ellipsis}, Tuple{Vararg{Int}}}
    )
    permblocks1 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[2])
    return (permblocks1, permblocks[2])
end
function to_permblocks(
        a, permblocks::Tuple{Tuple{Vararg{Int}}, Tuple{Ellipsis}}
    )
    permblocks2 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[1])
    return (permblocks[1], permblocks2)
end

function matricizeperm(a, perm_codomain, perm_domain)
    return matricizeperm(FusionStyle(a), a, perm_codomain, perm_domain)
end
function matricizeperm(
        style::FusionStyle, a, perm_codomain, perm_domain
    )
    return matricizeperm(style, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end

# ==================================  matricizeopperm  =====================================

"""
    matricizeopperm(op, a, perm_codomain, perm_domain)

Matricize `a` with element-wise operation `op` folded in. Returns a matrix representing
`op.(matricizeperm(a, perm_codomain, perm_domain))`.

Has "maybe alias" semantics: the result may be a view/wrapper aliasing `a` or a fresh
copy, depending on the fusion style and array type. The caller should treat the result
as read-only.
"""
function matricizeopperm(op, a, perm_codomain, perm_domain)
    return matricizeopperm(FusionStyle(a), op, a, perm_codomain, perm_domain)
end
function matricizeopperm(
        style::FusionStyle, op, a, perm_codomain, perm_domain
    )
    return matricizeopperm(style, op, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end
# Classifies how `matricize` realizes the bipermutation `(perm_codomain, perm_domain)`
# against storage, so `matricizeopperm` can skip the redundant permuted copy:
#   ReshapeMatricizeKind   — the groups are already in storage order, so the permute is a
#                            no-op and `matricize(style, a, ...)` can be called directly.
#                            For a dense array that is a `reshape` view; for a graded array
#                            it still gathers blocks, but skips the extra permute copy.
#   TransposeMatricizeKind — the only reordering is a codomain/domain swap, which a dense
#                            array realizes as a `transpose` of a `reshape` (a view gemm
#                            reads via BLAS' transpose flag).
#   PermuteMatricizeKind   — the groups interleave storage, so a permuted copy is required.
# Pure: depends only on the index pattern, not on `a`'s data. Dispatched on `FusionStyle`.
# The generic classifier only recognizes the always-safe `ReshapeMatricizeKind` (skipping a
# no-op permute is valid for any style); `TransposeMatricizeKind` is opt-in for styles whose
# `matricize` composes with a lazy `transpose`, currently only `ReshapeFusion`.
@enum MatricizeKind ReshapeMatricizeKind TransposeMatricizeKind PermuteMatricizeKind

# Whether `perm` is the identity permutation `(1, …, n)`.
isidentityperm(perm::Tuple{Vararg{Int}}) = perm == ntuple(identity, length(perm))

function matricizekind(
        ::FusionStyle, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    # Already in storage order: the permute is a no-op, so `matricize` can run directly.
    isidentityperm((perm_codomain..., perm_domain...)) && return ReshapeMatricizeKind
    return PermuteMatricizeKind
end

# Whether `matricizeperm(style, a, perm_codomain, perm_domain)` aliases `a` — returns a view (so a
# `mul!` into it writes through to `a`) rather than freshly allocated storage. Only a style whose
# `matricize` is itself a view (a dense reshape) can alias, and only when the bipermutation needs
# no permuted copy. Defaults to `false`: a style that gathers into new storage, such as a graded
# array, never aliases its input.
matricizepermaliases(::FusionStyle, perm_codomain, perm_domain) = false

# Skip the permuted copy when the classifier says it is unnecessary. `ReshapeMatricizeKind`
# calls `matricize` directly on `a` (a view for dense, a gather without the extra permute
# for graded); `TransposeMatricizeKind` returns a lazy `transpose` of the reshape. Both
# fast paths require `op === identity`, since a plain view cannot carry a fused `op` like
# `conj`. The result may alias `a` and must be treated as read-only, matching the docstring.
function matricizeopperm(
        style::FusionStyle, op, a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipermutation"))
    if op === identity
        kind = matricizekind(style, perm_codomain, perm_domain)
        kind == ReshapeMatricizeKind &&
            return matricize(style, a, Val(length(perm_codomain)))
        kind == TransposeMatricizeKind &&
            return transpose(matricize(style, a, Val(length(perm_domain))))
    end
    a_perm_op = permutedimsop(op, a, perm_codomain, perm_domain)
    return matricize(style, a_perm_op, Val(length(perm_codomain)))
end

# ====================================  unmatricize  =======================================
# Split form: `codomain_axes` and `domain_axes` are the destination axes for the codomain and
# domain groups, given codomain-facing (un-dualized), the same convention as `similar_map`. A
# fusion style stores the domain axes dualized, so its overload re-dualizes them with `conj`
# (a no-op on a dense axis). This is the primary overload point for new fusion styles.
# Permutation is handled separately by `unmatricizeperm`, so `unmatricize` never has to
# disambiguate axis tuples from permutation tuples regardless of how unconstrained `m` and the
# axes are.
function unmatricize(style::FusionStyle, m, codomain_axes, domain_axes)
    return throw(MethodError(unmatricize, (style, m, codomain_axes, domain_axes)))
end
function unmatricize(m, codomain_axes, domain_axes)
    return unmatricize(FusionStyle(m), m, codomain_axes, domain_axes)
end

# Split `axes` into its codomain and domain groups like `bipartition`, but present the domain
# group codomain-facing (un-dualized) with `conj`, the convention `unmatricize` and `similar_map`
# take. The domain axes `bipartition` reads off `axes(a)` are in the stored (dualized) form, so
# this bridges from `axes(a)` to the `unmatricize` axis convention (a no-op on dense axes).
function bipartition_axes(t::Tuple, split...)
    codomain_axes, domain_axes = bipartition(t, split...)
    return codomain_axes, conj.(domain_axes)
end

# Inverse-bipermutation form: split `axes_dest` into codomain/domain groups reordered by the
# inverse bipermutation, unmatricize in that order, then permute back.
function unmatricizeperm(
        m, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    return unmatricizeperm(FusionStyle(m), m, axes_dest, invperm_codomain, invperm_domain)
end
function unmatricizeperm(
        style::FusionStyle, m, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    length(axes_dest) == length(invbiperm) ||
        throw(ArgumentError("axes do not match permutation"))
    codomain_axes, domain_axes = bipartition_axes(axes_dest, invbiperm)
    a12 = unmatricize(style, m, codomain_axes, domain_axes)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(axes_dest)))
    return bipermutedims(a12, biperm_dest)
end

function unmatricizeperm!(
        a_dest, m,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    return unmatricizeperm!(FusionStyle(m), a_dest, m, invperm_codomain, invperm_domain)
end
function unmatricizeperm!(
        style::FusionStyle, a_dest, m,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    codomain_axes, domain_axes = bipartition_axes(axes(a_dest), invbiperm)
    a_perm = unmatricize(style, m, codomain_axes, domain_axes)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(axes(a_dest))))
    return bipermutedims!(a_dest, a_perm, biperm_dest)
end

# In-place split-axes counterpart of `unmatricize`, as `unmatricizeperm!` is of `unmatricizeperm`:
# scatter the fused matrix `m` back into `a_dest`'s existing storage across the codomain/domain
# split at `ndims_codomain`. The split applies no permutation, so this is `unmatricizeperm!` at the
# trivial bipermutation, reusing its in-place block scatter (no intermediate `unmatricize` copy).
function unmatricize!(a_dest, m, ndims_codomain::Val)
    K = unval(ndims_codomain)
    N = ndims(a_dest)
    return unmatricizeperm!(
        a_dest,
        m,
        ntuple(identity, Val(K)),
        ntuple(i -> K + i, Val(N - K))
    )
end

# Defaults to ReshapeFusion, a simple reshape
struct ReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:AbstractArray}) = ReshapeFusion()
function matricize(::ReshapeFusion, a, ndims_codomain::Val)
    unval(ndims_codomain) ≤ ndims(a) ||
        throw(ArgumentError("Codomain length exceeds number of dimensions."))
    size_codomain, size_domain = bipartition(size(a), ndims_codomain)
    return reshape(a, (prod(size_codomain), prod(size_domain)))
end
# A dense array additionally realizes a codomain/domain swap as a lazy `transpose` of a
# reshape (a view), so it opts into `TransposeMatricizeKind` on top of the generic
# reshape/permute classification.
function matricizekind(
        ::ReshapeFusion, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    isidentityperm((perm_codomain..., perm_domain...)) && return ReshapeMatricizeKind
    isidentityperm((perm_domain..., perm_codomain...)) && return TransposeMatricizeKind
    return PermuteMatricizeKind
end
# A dense reshape/transpose is a view of `a`; only a permuted copy detaches. So the matricized
# output aliases `a` for every kind except `PermuteMatricizeKind`.
function matricizepermaliases(style::ReshapeFusion, perm_codomain, perm_domain)
    return matricizekind(style, perm_codomain, perm_domain) != PermuteMatricizeKind
end
# A dense reshape ignores the codomain/domain split: it just reshapes to the concatenated axes.
# `conj` re-dualizes the codomain-facing `domain_axes` into stored form, a no-op on a dense axis.
function unmatricize(style::ReshapeFusion, m, codomain_axes, domain_axes)
    return reshape(m, (codomain_axes..., conj.(domain_axes)...))
end
