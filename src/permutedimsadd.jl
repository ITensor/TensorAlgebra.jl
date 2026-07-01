using Strided: Strided
using StridedViews: StridedViews as SV

"""
    permuteddims(a, perm)

Lazy `permutedims`. For an `AbstractArray` this is a `Base.PermutedDimsArray` view; for any
other operand it is a generic [`PermutedDims`](@ref) node. This is an extension hook:
downstream types can overload it to return their own lazy permuted-dims type.
"""
permuteddims(a::AbstractArray, perm) = PermutedDimsArray(a, perm)
permuteddims(a, perm) = PermutedDims(a, perm)

"""
    PermutedDims(parent, perm)

Lazy permuted-dims wrapper storing `parent` and the permutation `perm` in fields (unlike
`Base.PermutedDimsArray`, which encodes `perm` in a type parameter), so it constructs cheaply
from a runtime permutation. Primarily for internal use to track permutations in linear
broadcasting.
"""
struct PermutedDims{P, Perm}
    parent::P
    perm::Perm
end
Base.parent(a::PermutedDims) = a.parent

# `PermutedDims` is not an `AbstractArray`, so `Base.Broadcast.broadcastable` would otherwise try
# to `collect` it. These make it a valid `Broadcasted` leaf (shape = the parent's axes reordered
# by `perm`); it is never indexed, since the linear-broadcast fold absorbs it into a single
# `bipermutedimsopadd!` on `parent`. `axes`/`BroadcastStyle` are only reached when the parent is
# array-like (the only case that broadcasts); the direct `bipermutedimsopadd!` path never uses them.
Base.ndims(a::PermutedDims) = length(a.perm)
Base.axes(a::PermutedDims) = map(d -> axes(parent(a), d), a.perm)
Base.axes(a::PermutedDims, d::Int) = axes(a)[d]
Base.size(a::PermutedDims) = map(length, axes(a))
Base.eltype(a::PermutedDims) = eltype(parent(a))
Base.Broadcast.broadcastable(a::PermutedDims) = a
function Base.Broadcast.BroadcastStyle(::Type{<:PermutedDims{P}}) where {P}
    return Base.Broadcast.BroadcastStyle(P)
end

# ---------------------------------------------------------------------------- #
# bipermutedimsopadd! — the primary materialization primitive
# ---------------------------------------------------------------------------- #

function bipermutedimsopadd! end

# The destination holds `op.(src)` permuted, so its axes are the permuted source axes with
# `op` applied. `op` is restricted to `identity` and `conj` (see `bipermutedimsopadd!`), both
# of which act on axes: `conj` dualizes a graded axis (and is a no-op on a dense axis),
# `identity` leaves it unchanged.
function check_input(
        ::typeof(bipermutedimsopadd!), dest, op, src,
        perm_codomain, perm_domain
    )
    op === identity || op === conj ||
        throw(ArgumentError("`op` must be `identity` or `conj`, got `$op`"))
    perm = (perm_codomain..., perm_domain...)
    ndims(dest) == length(perm) ||
        throw(DimensionMismatch("destination ndims does not match permutation length"))
    axes(dest) == map(p -> op(axes(src, p)), perm) ||
        throw(DimensionMismatch("destination axes do not match permuted source axes"))
    return nothing
end

"""
    bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β)

`dest = β * dest + α * permutedims(op.(src), (perm_codomain..., perm_domain...))`.

This is the primary overload point for downstream array types that want to
implement op-aware bipartitioned permutation + accumulation (e.g., fuse `conj`
into the copy, or use lazy wrappers like `StridedView` with op metadata).

The `op` is the conjugation flag expressed as a function — `identity` or `conj`, analogous
to TensorOperations' boolean `conjA`/`conjB`. On graded axes `conj` dualizes; on dense axes
it is a no-op. Transposition/permutation is carried by the `perm` arguments, not by `op`.

The default implementation flattens the bipartitioned permutation, applies `op`
element-wise, permutes, then accumulates via broadcasting with Strided.jl
optimization when possible.
"""
function bipermutedimsopadd!(
        dest, op, src,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)
    check_input(bipermutedimsopadd!, dest, op, src, perm_codomain, perm_domain)

    dest′ = SV.StridedView(dest)
    src′ = permutedims(SV.StridedView(src), perm)
    _opadd!(dest′, op, src′, α, β)
    return dest
end

function _opadd!(dest::AbstractArray, op, src::AbstractArray, α, β)
    if op === identity
        if iszero(β)
            dest .= α .* src
        else
            dest .= β .* dest .+ α .* src
        end
    else
        if iszero(β)
            dest .= α .* op.(src)
        else
            dest .= β .* dest .+ α .* op.(src)
        end
    end
    return dest
end

_permuteddims_perm(::PermutedDimsArray{<:Any, <:Any, perm}) where {perm} = perm
_permuteddims_perm(a::PermutedDims) = a.perm

# Both the dense `PermutedDimsArray` view and the generic `PermutedDims` node absorb the same
# way: compose the wrapper's permutation `w` into the outer bipermutation and forward to a
# single leaf `bipermutedimsopadd!` on the parent.
function bipermutedimsopadd!(
        dest, op, src::Union{PermutedDimsArray, PermutedDims},
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    w = _permuteddims_perm(src)
    return bipermutedimsopadd!(
        dest, op, parent(src),
        map(j -> w[j], perm_codomain), map(j -> w[j], perm_domain),
        α, β
    )
end

# ---------------------------------------------------------------------------- #
# permutedimsopadd! — flat-permutation interface
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op.(src), perm)`.

This is the single materialization primitive for `LinearBroadcasted` types.
Downstream array types should implement `bipermutedimsopadd!` for the
bipartitioned permutation version; this flat-permutation overload forwards to it
with `perm_domain = ()`.
"""
function permutedimsopadd!(
        dest, op, src, perm, α::Number, β::Number
    )
    return bipermutedimsopadd!(dest, op, src, perm, (), α, β)
end

# ---------------------------------------------------------------------------- #
# Convenience functions that lower to permutedimsopadd!
# ---------------------------------------------------------------------------- #

"""
    permutedimsadd!(dest, src, perm, α, β)

`dest = β * dest + α * permutedims(src, perm)`.
"""
function permutedimsadd!(
        dest, src, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, identity, src, perm, α, β)
end

"""
    add!(dest, src, α, β)

`dest = β * dest + α * src`.
"""
function add!(dest, src, α::Number, β::Number)
    return permutedimsopadd!(dest, identity, src, ntuple(identity, ndims(src)), α, β)
end

"""
    add!(dest, src)

`dest .+= src`.
"""
add!(dest, src) = add!(dest, src, true, true)
