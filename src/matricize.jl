using EllipsisNotation: Ellipsis
using LinearAlgebra: Diagonal

# =====================================  MatricizeStyle  ======================================
abstract type MatricizeStyle end

MatricizeStyle(x) = MatricizeStyle(typeof(x))
MatricizeStyle(T::Type) = throw(MethodError(MatricizeStyle, (T,)))
MatricizeStyle(style1::Style, style2::Style) where {Style <: MatricizeStyle} = Style()
MatricizeStyle(style1::MatricizeStyle, style2::MatricizeStyle) = ReshapeMatricize()

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
    # Validate against `src` here: `bipermutedimsopadd!`'s `check_input` compares against `dest`,
    # which `allocate_output` builds from the same perms, so it cannot catch a non-covering perm.
    perm = (perm_codomain..., perm_domain...)
    (ndims(src) == length(perm) && isperm(perm)) ||
        throw(ArgumentError("Invalid bipermutation"))
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
# Copy convention: `bipermutedims`/`permutedims` always copy (Base `permutedims` semantics). At
# the trivial (`Val`) split the sharing story is exact: `matricizeview` shares `a`'s memory,
# `matricizecopy` returns fresh storage the caller owns, and `matricize` aliases `a` iff
# `ismatricizeview` — so a consumer that writes into a destination checks the trait and writes
# through `matricizeview`, and a consumer that mutates an input either owns a `matricizecopy`
# result by contract or materializes an owned matrix with `MatrixAlgebraKit.copy_input` (see the
# owned tier in `factorizations.jl`). `matricizeperm`/`matricizeopperm` keep maybe-alias
# semantics (the result may view or copy; treat it as read-only) until the planned op/perm-form
# trait lands, and `matricizeview` deliberately has no perm form pending that op/perm-layer
# design.

# `matricize` at the trivial split routes on the style's sharing declaration. Styles implement
# the three leaves (`ismatricizeview`, `matricizeview`, `matricizecopy`) rather than overloading
# `matricize` itself. This assumes the permutation was already performed.
function matricize(style::MatricizeStyle, a, ndims_codomain::Val)
    ismatricizeview(style, a, ndims_codomain) &&
        return matricizeview(style, a, ndims_codomain)
    return matricizecopy(style, a, ndims_codomain)
end
function matricize(a, ndims_codomain::Val)
    return matricize(MatricizeStyle(a), a, ndims_codomain)
end

# Partial: defined only where `ismatricizeview` is `true`, and always returns a matricization
# sharing `a`'s memory (the `StridedView` partial-constructor pattern).
function matricizeview(style::MatricizeStyle, a, ndims_codomain::Val)
    return throw(MethodError(matricizeview, (style, a, ndims_codomain)))
end
# Total: always returns a matricization in fresh storage the caller owns.
function matricizecopy(style::MatricizeStyle, a, ndims_codomain::Val)
    return throw(MethodError(matricizecopy, (style, a, ndims_codomain)))
end

# `bipermutedims` always copies and `matricize` might return a view, so the result is
# guaranteed to be a copy.
function matricizecopy(
        style::MatricizeStyle, a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    a_perm = bipermutedims(a, perm_codomain, perm_domain)
    return matricize(style, a_perm, Val(length(perm_codomain)))
end

function matricizeperm(
        a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    return matricizeperm(MatricizeStyle(a), a, perm_codomain, perm_domain)
end
# Thin wrapper around `matricizeopperm` with identity op — the actual matricization logic
# (and the matricize-style overload point for folding ops into matricization) lives in
# `matricizeopperm`.
function matricizeperm(
        style::MatricizeStyle, a,
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
    return matricizeperm(MatricizeStyle(a), a, perm_codomain, perm_domain)
end
function matricizeperm(
        style::MatricizeStyle, a, perm_codomain, perm_domain
    )
    return matricizeperm(style, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end

# ==================================  matricizeopperm  =====================================

"""
    matricizeopperm(op, a, perm_codomain, perm_domain)

Matricize `a` with element-wise operation `op` folded in. Returns a matrix representing
`op.(matricizeperm(a, perm_codomain, perm_domain))`.

Has "maybe alias" semantics: the result may be a view/wrapper aliasing `a` or a fresh
copy, depending on the matricize style and array type. The caller should treat the result
as read-only.
"""
function matricizeopperm(op, a, perm_codomain, perm_domain)
    return matricizeopperm(MatricizeStyle(a), op, a, perm_codomain, perm_domain)
end
function matricizeopperm(
        style::MatricizeStyle, op, a, perm_codomain, perm_domain
    )
    return matricizeopperm(style, op, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end
# Whether `perm` is the identity permutation `(1, …, n)`.
isidentityperm(perm::Tuple{Vararg{Int}}) = perm == ntuple(identity, length(perm))

# The identity bipermutation is a no-op permute, so `matricize` runs directly on `a` (a view
# for dense, a gather without the extra permute copy for graded); the fast path requires
# `op === identity`, since a plain view cannot carry a fused `op` like `conj`. The result may
# alias `a` and must be treated as read-only, matching the docstring.
function matricizeopperm(
        style::MatricizeStyle, op, a,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipermutation"))
    op === identity && isidentityperm((perm_codomain..., perm_domain...)) &&
        return matricize(style, a, Val(length(perm_codomain)))
    a_perm_op = permutedimsop(op, a, perm_codomain, perm_domain)
    return matricize(style, a_perm_op, Val(length(perm_codomain)))
end

# ==================================  ismatricizeview  =====================================
# `true` iff `matricize(style, a, ndims_codomain)` shares `a`'s memory (writes to it are writes
# to `a`) — the `isstrided`/`StridedView` pattern (also TensorKit's `has_shared_permute` and
# TensorOperations' `isblasdestination`). Styles overload the `Val` (trivial split) form to
# declare which splits share; the bipermutation form delegates to it at the identity and is
# `false` (fail-safe) everywhere else. A general `ismatricizeview(style, op, a, perm_codomain,
# perm_domain)` form (op and bipermutation view-sets) is planned; these are its
# `op === identity` special cases.
ismatricizeview(::MatricizeStyle, a, ndims_codomain::Val) = false
function ismatricizeview(
        style::MatricizeStyle, a,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    isidentityperm((invperm_codomain..., invperm_domain...)) || return false
    return ismatricizeview(style, a, Val(length(invperm_codomain)))
end

# ====================================  unmatricize  =======================================
# Split form: `axes_codomain` and `axes_domain` are the destination axes for the codomain and
# domain groups, given codomain-facing (un-dualized), the same convention as `similar_map`. A
# matricize style stores the domain axes dualized, so its overload re-dualizes them with `conj`
# (a no-op on a dense axis). This is the primary overload point for new matricize styles.
# Permutation is handled separately by `unmatricizeperm`, so `unmatricize` never has to
# disambiguate axis tuples from permutation tuples regardless of how unconstrained `m` and the
# axes are.
function unmatricize(style::MatricizeStyle, m, axes_codomain, axes_domain)
    return throw(MethodError(unmatricize, (style, m, axes_codomain, axes_domain)))
end
function unmatricize(m, axes_codomain, axes_domain)
    return unmatricize(MatricizeStyle(m), m, axes_codomain, axes_domain)
end

# Split `axes` into its codomain and domain groups like `bipartition`, but present the domain
# group codomain-facing (un-dualized) with `conj`, the convention `unmatricize` and `similar_map`
# take. The domain axes `bipartition` reads off `axes(a)` are in the stored (dualized) form, so
# this bridges from `axes(a)` to the `unmatricize` axis convention (a no-op on dense axes).
function bipartition_axes(t::Tuple, split...)
    axes_codomain, axes_domain = bipartition(t, split...)
    return axes_codomain, conj.(axes_domain)
end

# Inverse-bipermutation form: split `axes_dest` into codomain/domain groups reordered by the
# inverse bipermutation, unmatricize in that order, then permute back.
function unmatricizeperm(
        m, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    return unmatricizeperm(
        MatricizeStyle(m),
        m,
        axes_dest,
        invperm_codomain,
        invperm_domain
    )
end
function unmatricizeperm(
        style::MatricizeStyle, m, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    length(axes_dest) == length(invbiperm) ||
        throw(ArgumentError("axes do not match permutation"))
    axes_codomain, axes_domain = bipartition_axes(axes_dest, invbiperm)
    a12 = unmatricize(style, m, axes_codomain, axes_domain)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(invbiperm)))
    return bipermutedims(a12, biperm_dest)
end

function unmatricizeperm!(
        a_dest, m,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    return unmatricizeperm!(MatricizeStyle(m), a_dest, m, invperm_codomain, invperm_domain)
end
function unmatricizeperm!(
        style::MatricizeStyle, a_dest, m,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    axes_codomain, axes_domain = bipartition_axes(axes(a_dest), invbiperm)
    a_perm = unmatricize(style, m, axes_codomain, axes_domain)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(invbiperm)))
    return bipermutedims!(a_dest, a_perm, biperm_dest)
end

# In-place split-axes counterpart of `unmatricize`, as `unmatricizeperm!` is of `unmatricizeperm`:
# scatter the fused matrix `m` back into `a_dest`'s existing storage across the codomain/domain
# split at `ndims_codomain`. The split applies no permutation, so this is `unmatricizeperm!` at the
# trivial bipermutation, reusing its in-place block scatter (no intermediate `unmatricize` copy).
function unmatricize!(style::MatricizeStyle, a_dest, m, ndims_codomain::Val)
    K = unval(ndims_codomain)
    N = ndims(a_dest)
    return unmatricizeperm!(
        style,
        a_dest,
        m,
        ntuple(identity, Val(K)),
        ntuple(i -> K + i, Val(N - K))
    )
end
function unmatricize!(a_dest, m, ndims_codomain::Val)
    return unmatricize!(MatricizeStyle(a_dest), a_dest, m, ndims_codomain)
end

# Defaults to ReshapeMatricize, a simple reshape
struct ReshapeMatricize <: MatricizeStyle end
MatricizeStyle(::Type{<:AbstractArray}) = ReshapeMatricize()
# A dense reshape matricization is a lazy wrapper at any split, so it always shares memory.
ismatricizeview(::ReshapeMatricize, a, ndims_codomain::Val) = true
function matricizeview(::ReshapeMatricize, a, ndims_codomain::Val)
    unval(ndims_codomain) ≤ ndims(a) ||
        throw(ArgumentError("Codomain length exceeds number of dimensions."))
    size_codomain, size_domain = bipartition(size(a), ndims_codomain)
    return reshape(a, (prod(size_codomain), prod(size_domain)))
end
function matricizecopy(style::ReshapeMatricize, a, ndims_codomain::Val)
    return copy(matricizeview(style, a, ndims_codomain))
end
# The matricized input's rows must be the fused codomain and its columns the fused domain.
# `reshape` alone only checks the total element count, so a wrong split with the right total
# would reshape silently.
function check_input(::typeof(unmatricize), m, axes_codomain, axes_domain)
    (
        ndims(m) == 2 &&
            size(m, 1) == prod(length, axes_codomain; init = 1) &&
            size(m, 2) == prod(length, axes_domain; init = 1)
    ) || throw(DimensionMismatch("`unmatricize` axes do not match the matrix size"))
    return nothing
end
# A dense reshape ignores the codomain/domain split: it just reshapes to the concatenated axes.
# `conj` re-dualizes the codomain-facing `axes_domain` into stored form, a no-op on a dense axis.
function unmatricize(style::ReshapeMatricize, m, axes_codomain, axes_domain)
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return reshape(m, (axes_codomain..., conj.(axes_domain)...))
end
