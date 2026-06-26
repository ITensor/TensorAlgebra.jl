using EllipsisNotation: Ellipsis
using LinearAlgebra: Diagonal

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

FusionStyle(x) = FusionStyle(typeof(x))
FusionStyle(T::Type) = throw(MethodError(FusionStyle, (T,)))
FusionStyle(style1::Style, style2::Style) where {Style <: FusionStyle} = Style()
FusionStyle(style1::FusionStyle, style2::FusionStyle) = ReshapeFusion()

# =======================================  misc  ========================================
function trivial_axis(
        style::FusionStyle, side::Val{:codomain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return throw(MethodError(trivial_axis, (style, side, a, axes_codomain, axes_domain)))
end
function trivial_axis(
        style::FusionStyle, ::Val{:domain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_axis(style, Val(:codomain), a, axes_codomain, axes_domain)
end
function trivial_axis(
        style::FusionStyle, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_axis(style, Val(:codomain), a, axes_codomain, axes_domain)
end
function trivial_axis(style::FusionStyle, a::AbstractArray)
    return trivial_axis(style, a, (), ())
end
function trivial_axis(
        a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_axis(FusionStyle(a), a, axes_codomain, axes_domain)
end
function trivial_axis(side::Val, a::AbstractArray)
    return trivial_axis(FusionStyle(a), side, a)
end
function trivial_axis(a::AbstractArray)
    return trivial_axis(FusionStyle(a), a)
end

# Tensor product two spaces (ranges) together based on a fusion style.
function tensor_product_axis(
        style::FusionStyle, side::Val{:codomain},
        r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    return throw(MethodError(tensor_product_axis, (style, side, r1, r2)))
end
function tensor_product_axis(
        style::FusionStyle, ::Val{:domain}, r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    return tensor_product_axis(style, Val(:codomain), r1, r2)
end
function tensor_product_axis(
        style::FusionStyle, r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    return tensor_product_axis(style, Val(:codomain), r1, r2)
end
function tensor_product_axis(side::Val, r1::AbstractUnitRange, r2::AbstractUnitRange)
    style = tensor_product_fusionstyle(r1, r2)
    return tensor_product_axis(style, side, r1, r2)
end
function tensor_product_axis(r1::AbstractUnitRange, r2::AbstractUnitRange)
    style = tensor_product_fusionstyle(r1, r2)
    return tensor_product_axis(style, r1, r2)
end
function tensor_product_fusionstyle(r1::AbstractUnitRange, r2::AbstractUnitRange)
    return FusionStyle(FusionStyle(r1), FusionStyle(r2))
end

"""
    TensorAlgebra.trivialrange(R::Type{<:AbstractUnitRange})
    TensorAlgebra.trivialrange(r::AbstractUnitRange)

Return the identity range for `tensor_product_axis` on ranges of type `R`,
i.e. a one-dimensional range `t` for which fusing `t` with any other range
of the same family leaves that range unchanged. Defaults to `Base.OneTo(1)`;
downstream packages overload the type-level method to return their own
identity (for example, a charge-0 one-dimensional sector for a graded range).
"""
trivialrange(r::AbstractUnitRange) = trivialrange(typeof(r))
trivialrange(::Type{<:AbstractUnitRange}) = Base.OneTo(1)

function fused_axis(
        style::FusionStyle, side::Val{:codomain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    init_axis = trivial_axis(style, side, a, axes_codomain, axes_domain)
    return reduce(axes_codomain; init = init_axis) do ax1, ax2
        return tensor_product_axis(style, side, ax1, ax2)
    end
end
function fused_axis(
        style::FusionStyle, side::Val{:domain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    init_axis = trivial_axis(style, side, a, axes_codomain, axes_domain)
    return reduce(axes_domain; init = init_axis) do ax1, ax2
        return tensor_product_axis(style, side, ax1, ax2)
    end
end
function matricize_axes(
        style::FusionStyle, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    axis_codomain = fused_axis(style, Val(:codomain), a, axes_codomain, axes_domain)
    axis_domain = fused_axis(style, Val(:domain), a, axes_codomain, axes_domain)
    return axis_codomain, axis_domain
end
function matricize_axes(
        a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return matricize_axes(FusionStyle(a), a, axes_codomain, axes_domain)
end
function matricize_axes(style::FusionStyle, a::AbstractArray, ndims_codomain::Val)
    unval(ndims_codomain) ≤ ndims(a) ||
        throw(ArgumentError("Codomain length exceeds number of dimensions."))
    return matricize_axes(style, a, bipartition(axes(a), ndims_codomain)...)
end
function matricize_axes(a::AbstractArray, ndims_codomain::Val)
    return matricize_axes(FusionStyle(a), a, ndims_codomain)
end

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

function allocate_output(::typeof(permutedimsop), op, src::AbstractArray, perm_co, perm_do)
    T = Base.promote_op(op, eltype(src))
    axes_co = map(i -> axes(src, i), perm_co)
    axes_do = map(i -> axes(src, i), perm_do)
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

function unmatricizeadd!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
        α::Number, β::Number
    )
    return unmatricizeadd!(
        FusionStyle(a_dest), a_dest, m, invperm_codomain, invperm_domain, α, β
    )
end
function unmatricizeadd!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
        α::Number, β::Number
    )
    invbiperm = BiTuple(invperm_codomain, invperm_domain)
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    # Reshape `m` to the destination's matricized axes (a view), then permute it
    # straight into `a_dest` with accumulation in a single pass, rather than
    # allocating a permuted copy and then adding it. Mirrors `unmatricize!`.
    a_perm = unmatricize(style, m, bipartition(axes(a_dest), invbiperm)...)
    biperm_dest = BiTuple(Tuple(invperm(invbiperm)), Val(length_codomain(axes(a_dest))))
    return bipermutedimsopadd!(
        a_dest, identity, a_perm, biperm_dest.t1, biperm_dest.t2, α, β
    )
end

# Defaults to ReshapeFusion, a simple reshape
struct ReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:AbstractArray}) = ReshapeFusion()
function trivial_axis(
        style::ReshapeFusion, side::Val{:codomain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return Base.OneTo(1)
end
function tensor_product_axis(
        style::ReshapeFusion, side::Val{:codomain},
        r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    return Base.OneTo(length(r1) * length(r2))
end
function matricize(style::ReshapeFusion, a::AbstractArray, ndims_codomain::Val)
    return reshape(a, matricize_axes(style, a, ndims_codomain))
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
