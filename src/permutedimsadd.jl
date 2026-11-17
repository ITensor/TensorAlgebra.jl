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
# `bipermutedimsconjadd!` on `parent`. `axes`/`BroadcastStyle` are only reached when the parent is
# array-like (the only case that broadcasts); the direct `bipermutedimsconjadd!` path never uses them.
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
# bipermutedimsconjadd! — the primary materialization primitive
# ---------------------------------------------------------------------------- #

function bipermutedimsconjadd! end

# The destination holds the permuted source, conjugated when `isconj` is set, so its axes are the
# permuted source axes, dualized under `isconj` (a no-op on a dense axis).
function check_input(
        ::typeof(bipermutedimsconjadd!), dest, isconj::Bool, src,
        perm_codomain, perm_domain
    )
    perm = (perm_codomain..., perm_domain...)
    ndims(dest) == length(perm) ||
        throw(DimensionMismatch("destination ndims does not match permutation length"))
    axes_src = map(p -> axes(src, p), perm)
    axes(dest) == (isconj ? map(conj, axes_src) : axes_src) ||
        throw(DimensionMismatch("destination axes do not match permuted source axes"))
    return nothing
end

"""
    bipermutedimsconjadd!(dest, isconj::Bool, src, perm_codomain, perm_domain, α, β)

`dest = β * dest + α * permutedims(src, (perm_codomain..., perm_domain...))`, with `src`
conjugated when `isconj` is `true`.

This is the overload point for downstream array types that implement bipartitioned permutation
and accumulation. `isconj` is TensorOperations' `conjA` flag: on graded axes it dualizes, on
dense axes it is a no-op. Callers usually go through [`bipermutedimsopadd!`](@ref), which
forwards here.

The default implementation permutes lazily and accumulates by broadcasting, through Strided.jl
when both operands are strided.
"""
function bipermutedimsconjadd!(
        dest, isconj::Bool, src,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)
    check_input(bipermutedimsconjadd!, dest, isconj, src, perm_codomain, perm_domain)
    if SV.isstrided(dest) && SV.isstrided(src)
        permutedimsconjadd_strided!(
            SV.StridedView(dest), isconj, SV.StridedView(src), perm, α, β
        )
    else
        # Non-strided operands (e.g. structured matrices like `Diagonal`) accumulate
        # through generic broadcasting.
        _conjadd!(dest, isconj, permuteddims(src, perm), α, β)
    end
    return dest
end

# The single entry into Strided's kernel for a permuted add. Its argument types are fixed by the
# element types and rank (the permutation, the conjugation flag and the scalars are run-time
# values), so it compiles once per element type and rank and is precompiled for the common ones.
function permutedimsconjadd_strided!(
        dest::SV.StridedView{T, N}, isconj::Bool, src::SV.StridedView{<:Any, N},
        perm::NTuple{N, Int}, α::Number, β::Number
    ) where {T, N}
    any(iszero, size(dest)) && return dest
    src_perm = Base.permutedims(src, perm)
    f = ConjScale(isconj, convert(T, α))
    if iszero(β)
        Strided._mapreducedim!(f, nothing, nothing, size(dest), (dest, src_perm))
    else
        # A run-time dispatch, so the accumulating kernel only compiles when it is used.
        Base.inferencebarrier(_permutedimsconjaccumulate_strided!)(
            dest, f, src_perm, convert(T, β)
        )
    end
    return dest
end
function _permutedimsconjaccumulate_strided!(dest, f, src_perm, β)
    Strided._mapreducedim!(f, +, Scale(β), size(dest), (dest, src_perm))
    return nothing
end

# `x -> α * x`, or `α * conj(x)` under the flag, as concrete callables for Strided's kernel.
struct ConjScale{T}
    isconj::Bool
    α::T
end
(f::ConjScale)(x) = f.α * (f.isconj ? conj(x) : x)
struct Scale{T}
    α::T
end
(f::Scale)(x) = f.α * x

for T in (Float64, ComplexF64), N in 1:6
    SVT = SV.StridedView{T, N, Memory{T}, typeof(identity)}
    precompile(permutedimsconjadd_strided!, (SVT, Bool, SVT, NTuple{N, Int}, Bool, Bool))
    precompile(_permutedimsconjaccumulate_strided!, (SVT, ConjScale{T}, SVT, T))
end

# Conjugates under a run-time flag, so a single broadcast kernel serves both cases.
struct ConjIf
    isconj::Bool
end
(f::ConjIf)(x) = f.isconj ? conj(x) : x

function _conjadd!(dest::AbstractArray, isconj::Bool, src::AbstractArray, α, β)
    f = ConjIf(isconj)
    if iszero(β)
        dest .= α .* f.(src)
    else
        # Permutes, contractions and factorizations almost always overwrite (`β = 0`). A run-time
        # dispatch here means the accumulating kernel only compiles when it is used.
        Base.inferencebarrier(_conjaccumulate!)(dest, f, src, α, β)
    end
    return dest
end
_conjaccumulate!(dest, f, src, α, β) = (dest .= β .* dest .+ α .* f.(src); nothing)

_permuteddims_perm(::PermutedDimsArray{<:Any, <:Any, perm}) where {perm} = perm
_permuteddims_perm(a::PermutedDims) = a.perm

# Both the dense `PermutedDimsArray` view and the generic `PermutedDims` node absorb the same
# way: compose the wrapper's permutation `w` into the outer bipermutation and forward to a
# single leaf `bipermutedimsconjadd!` on the parent.
function bipermutedimsconjadd!(
        dest, isconj::Bool, src::Union{PermutedDimsArray, PermutedDims},
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    w = _permuteddims_perm(src)
    return bipermutedimsconjadd!(
        dest, isconj, parent(src),
        map(j -> w[j], perm_codomain), map(j -> w[j], perm_domain),
        α, β
    )
end

"""
    bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β)

`dest = β * dest + α * permutedims(op.(src), (perm_codomain..., perm_domain...))`, where `op` is
`identity` or `conj`. Forwards to [`bipermutedimsconjadd!`](@ref) with `op === conj`, which is
the method to overload.
"""
function bipermutedimsopadd!(
        dest, op, src, perm_codomain, perm_domain, α::Number, β::Number
    )
    return bipermutedimsconjadd!(dest, _isconj(op), src, perm_codomain, perm_domain, α, β)
end

# The conjugation flag of an operation that the `op` forms accept.
function _isconj(op)
    op === identity || op === conj ||
        throw(ArgumentError("`op` must be `identity` or `conj`, got `$op`"))
    return op === conj
end

# ---------------------------------------------------------------------------- #
# permutedimsopadd! — flat-permutation interface
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op.(src), perm)`.

This is the single materialization primitive for `LinearBroadcasted` types.
Downstream array types should implement `bipermutedimsconjadd!` for the
bipartitioned permutation version. Forwards to [`permutedimsconjadd!`](@ref).
"""
function permutedimsopadd!(dest, op, src, perm, α::Number, β::Number)
    return permutedimsconjadd!(dest, _isconj(op), src, perm, α, β)
end
"""
    permutedimsconjadd!(dest, isconj::Bool, src, perm, α, β)

Like [`permutedimsopadd!`](@ref), with the operation given as a conjugation flag. Forwards to
`bipermutedimsconjadd!` with `perm_domain = ()`.
"""
function permutedimsconjadd!(dest, isconj::Bool, src, perm, α::Number, β::Number)
    return bipermutedimsconjadd!(dest, isconj, src, perm, (), α, β)
end

# ---------------------------------------------------------------------------- #
# Convenience functions that lower to permutedimsconjadd!
# ---------------------------------------------------------------------------- #

"""
    permutedimsadd!(dest, src, perm, α, β)

`dest = β * dest + α * permutedims(src, perm)`.
"""
function permutedimsadd!(
        dest, src, perm, α::Number, β::Number
    )
    return permutedimsconjadd!(dest, false, src, perm, α, β)
end

"""
    scaleadd!(dest, src, α, β)

`dest = β * dest + α * src`.
"""
function scaleadd!(dest, src, α::Number, β::Number)
    return permutedimsconjadd!(dest, false, src, ntuple(identity, ndims(src)), α, β)
end

"""
    add!(dest, src)

`dest .+= src`.
"""
add!(dest, src) = scaleadd!(dest, src, true, true)

# ---------------------------------------------------------------------------- #
# permutedims — out-of-place, optional bipartition
# ---------------------------------------------------------------------------- #

"""
    permutedims!(dest, a, perm)
    permutedims!(dest, a, perm_codomain, perm_domain)

In-place counterpart of [`permutedims`](@ref): write the permuted `a` into `dest`.
Both forms forward to `bipermutedimsconjadd!` with `α, β = true, false`; the flat form
passes an empty domain permutation.
"""
function permutedims!(dest, a, perm)
    return bipermutedimsconjadd!(dest, false, a, perm, (), true, false)
end
function permutedims!(dest, a, perm_codomain, perm_domain)
    return bipermutedimsconjadd!(dest, false, a, perm_codomain, perm_domain, true, false)
end

"""
    permutedims(a, perm)
    permutedims(a, perm_codomain, perm_domain)

Out-of-place permutation of `a`, mirroring `TensorKit.permute`. The single-permutation
form reorders every dimension into `perm`, giving an all-codomain result. The
two-permutation form additionally splits the dimensions into a codomain/domain
bipartition, with `perm_codomain` selecting the codomain dimensions and `perm_domain`
the domain ones.

Allocates the destination with [`similar_map`](@ref) and materializes it through
[`permutedims!`](@ref), so any operand implementing the `permutedimsconjadd!` /
`bipermutedimsconjadd!` interface (a dense array, a graded array, a `TensorMap`) is
permuted with no dedicated method. A dense operand ignores the bipartition and stores
the result flat.
"""
function permutedims(a, perm)
    dest = similar_map(a, eltype(a), map(p -> axes(a, p), perm), ())
    return permutedims!(dest, a, perm)
end
function permutedims(a, perm_codomain, perm_domain)
    axes_codomain = map(p -> axes(a, p), perm_codomain)
    # `similar_map` dualizes the domain axes it is given (as it does for `similar_map`
    # itself), so pass them pre-conjugated to land back on the source's own axes.
    axes_domain = map(p -> conj(axes(a, p)), perm_domain)
    dest = similar_map(a, eltype(a), axes_codomain, axes_domain)
    return permutedims!(dest, a, perm_codomain, perm_domain)
end
