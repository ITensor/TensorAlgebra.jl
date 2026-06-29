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
    TensorAlgebra.trivialrange(R::Type{<:AbstractUnitRange})
    TensorAlgebra.trivialrange(r::AbstractUnitRange)

Return the identity range for fusing ranges of type `R`: a one-dimensional range
`t` for which fusing `t` with any other range of the same family leaves that range
unchanged. Defaults to `Base.OneTo(1)`. Downstream packages overload the type-level
method to return their own identity (for example, a charge-0 one-dimensional sector
for a graded range).
"""
trivialrange(r::AbstractUnitRange) = trivialrange(typeof(r))
trivialrange(::Type{<:AbstractUnitRange}) = Base.OneTo(1)

# Default similar with bipartitioned axes: flatten to a plain tuple of axes.
# Downstream types (e.g., FusionTensor) can override to preserve bipartition.
function Base.similar(a::AbstractArray, T::Type, axes::BiTuple)
    return similar(a, T, Tuple(axes))
end

"""
    permutedimsop(op, src, perm_codomain, perm_domain)

Non-mutating version of `bipermutedimsopadd!`: returns
`op.(permutedims(src, (perm_codomain..., perm_domain...)))`.
"""
function permutedimsop(op, src::AbstractArray, perm_codomain, perm_domain)
    dest = allocate_output(permutedimsop, op, src, perm_codomain, perm_domain)
    return bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, true, false)
end

# The output holds `op.(src)` permuted, so `op` applies to the axes too: `conj` dualizes a
# graded axis (a no-op on a dense axis), `identity` leaves it unchanged, keeping axes and
# data in sync.
function allocate_output(::typeof(permutedimsop), op, src::AbstractArray, perm_co, perm_do)
    T = Base.promote_op(op, eltype(src))
    axes_co = map(i -> op(axes(src, i)), perm_co)
    axes_do = map(i -> op(axes(src, i)), perm_do)
    return similar(src, T, BiTuple(axes_co, axes_do))
end

function bipermutedims(a::AbstractArray, perm1, perm2)
    return permutedimsop(identity, a, perm1, perm2)
end
function bipermutedims!(a_dest::AbstractArray, a_src::AbstractArray, perm1, perm2)
    return bipermutedimsopadd!(a_dest, identity, a_src, perm1, perm2, true, false)
end
function bipermutedims(a::AbstractArray, biperm::BiTuple)
    return bipermutedims(a, biperm.t1, biperm.t2)
end
function bipermutedims!(
        a_dest::AbstractArray, a_src::AbstractArray, biperm::BiTuple
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
        style::FusionStyle, a::AbstractArray, ndims_codomain::Val
    )
    return throw(MethodError(matricize, (style, a, ndims_codomain)))
end
function matricize(a::AbstractArray, ndims_codomain::Val)
    return matricize(FusionStyle(a), a, ndims_codomain)
end

function matricize(
        a::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    return matricize(FusionStyle(a), a, perm_codomain, perm_domain)
end
# Thin wrapper around `matricizeop` with identity op — the actual matricization logic
# (and the fusion-style overload point for folding ops into matricization) lives in
# `matricizeop`.
function matricize(
        style::FusionStyle, a::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    return matricizeop(style, identity, a, perm_codomain, perm_domain)
end

# Process inputs such as `EllipsisNotation.Ellipsis`.
function to_permblocks(a::AbstractArray, permblocks::NTuple{2, Tuple{Vararg{Int}}})
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
        a::AbstractArray, permblocks::Tuple{Tuple{Ellipsis}, Tuple{Vararg{Int}}}
    )
    permblocks1 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[2])
    return (permblocks1, permblocks[2])
end
function to_permblocks(
        a::AbstractArray, permblocks::Tuple{Tuple{Vararg{Int}}, Tuple{Ellipsis}}
    )
    permblocks2 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[1])
    return (permblocks[1], permblocks2)
end

function matricize(a::AbstractArray, perm_codomain, perm_domain)
    return matricize(FusionStyle(a), a, perm_codomain, perm_domain)
end
function matricize(
        style::FusionStyle, a::AbstractArray, perm_codomain, perm_domain
    )
    return matricize(style, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end

# ====================================  matricizeop  =======================================

"""
    matricizeop(op, a, perm_codomain, perm_domain)

Matricize `a` with element-wise operation `op` folded in. Returns a matrix representing
`op.(matricize(a, perm_codomain, perm_domain))`.

Has "maybe alias" semantics: the result may be a view/wrapper aliasing `a` or a fresh
copy, depending on the fusion style and array type. The caller should treat the result
as read-only.
"""
function matricizeop(op, a::AbstractArray, perm_codomain, perm_domain)
    return matricizeop(FusionStyle(a), op, a, perm_codomain, perm_domain)
end
function matricizeop(
        style::FusionStyle, op, a::AbstractArray, perm_codomain, perm_domain
    )
    return matricizeop(style, op, a, to_permblocks(a, (perm_codomain, perm_domain))...)
end
# Classifies how `matricize` realizes the bipermutation `(perm_codomain, perm_domain)`
# against storage, so `matricizeop` can skip the redundant permuted copy:
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

# Skip the permuted copy when the classifier says it is unnecessary. `ReshapeMatricizeKind`
# calls `matricize` directly on `a` (a view for dense, a gather without the extra permute
# for graded); `TransposeMatricizeKind` returns a lazy `transpose` of the reshape. Both
# fast paths require `op === identity`, since a plain view cannot carry a fused `op` like
# `conj`. The result may alias `a` and must be treated as read-only, matching the docstring.
function matricizeop(
        style::FusionStyle, op, a::AbstractArray,
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
# This is the primary function that should be overloaded for new fusion styles.
function unmatricize(
        style::FusionStyle, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return throw(MethodError(unmatricize, (style, m, axes_codomain, axes_domain)))
end
function unmatricize(
        m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return unmatricize(FusionStyle(m), m, axes_codomain, axes_domain)
end

function unmatricize(
        m::AbstractMatrix, axes_dest,
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}}
    )
    return unmatricize(FusionStyle(m), m, axes_dest, invperm1, invperm2)
end
function unmatricize(
        style::FusionStyle, m::AbstractMatrix, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    length(axes_dest) == length(invbiperm) ||
        throw(ArgumentError("axes do not match permutation"))
    a12 = unmatricize(style, m, bipartition(axes_dest, invbiperm)...)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(axes_dest)))
    return bipermutedims(a12, biperm_dest)
end

function unmatricize!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    return unmatricize!(FusionStyle(m), a_dest, m, invperm_codomain, invperm_domain)
end
function unmatricize!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    a_perm = unmatricize(style, m, bipartition(axes(a_dest), invbiperm)...)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(axes(a_dest))))
    return bipermutedims!(a_dest, a_perm, biperm_dest)
end

# Defaults to ReshapeFusion, a simple reshape
struct ReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:AbstractArray}) = ReshapeFusion()
function matricize(::ReshapeFusion, a::AbstractArray, ndims_codomain::Val)
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
function unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return reshape(m, (axes_codomain..., axes_domain...))
end
