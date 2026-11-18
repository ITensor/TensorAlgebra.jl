using Strided: Strided
using StridedViews: StridedViews as SV

"""
    permuteddims(a, perm)

Lazy `permutedims`. For an `AbstractArray` this is a `Base.PermutedDimsArray` view, and for any
other operand it is a [`PermutedDims`](@ref) wrapper.
"""
permuteddims(a::AbstractArray, perm) = PermutedDimsArray(a, perm)
permuteddims(a, perm) = PermutedDims(a, perm)

"""
    PermutedDims(parent, perm)

Lazy permuted-dims wrapper of `parent` by the permutation `perm`. Unlike
`Base.PermutedDimsArray`, `perm` is stored in a field rather than a type parameter, so it
constructs cheaply from a runtime permutation.
"""
struct PermutedDims{P, Perm}
    parent::P
    perm::Perm
end
Base.parent(a::PermutedDims) = a.parent

# `PermutedDims` is not an `AbstractArray`, so these make it a valid broadcast leaf.
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
# bipermutedimsconjadd!
# ---------------------------------------------------------------------------- #

function bipermutedimsconjadd! end

"""
    check_input(f, args...)

Throw an error if `args` are not valid arguments to `f`. An array type's own method of `f` can
call it to validate its arguments.
"""
function check_input end

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
    bipermutedimsconjadd!(dest, isconj::Bool, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(src, (perm_codomain..., perm_domain...))`, with `src`
conjugated when `isconj` is `true`. Array types overload this function to customize it.

See also [`bipermutedimsopadd!`](@ref).
"""
function bipermutedimsconjadd!(
        dest, isconj::Bool, src, perm_codomain, perm_domain, α::Number, β::Number
    )
    return _bipermutedimsconjadd!(dest, isconj, src, perm_codomain, perm_domain, α, β)
end

# Lazy wrappers are unwrapped by dispatching on this internal function rather than on
# `bipermutedimsconjadd!`, so they never make an array type's own method ambiguous.
function _bipermutedimsconjadd!(
        dest, isconj::Bool, src, perm_codomain, perm_domain, α::Number, β::Number
    )
    return permutedimsconjadd_flat!(
        dest, isconj, src, (perm_codomain..., perm_domain...), α, β
    )
end
function _bipermutedimsconjadd!(
        dest, isconj::Bool, src::Union{PermutedDimsArray, PermutedDims},
        perm_codomain, perm_domain, α::Number, β::Number
    )
    w = permuteddims_perm(src)
    return bipermutedimsconjadd!(
        dest, isconj, parent(src),
        map(j -> w[j], perm_codomain), map(j -> w[j], perm_domain), α, β
    )
end
# The work only depends on the flat permutation, so this is kept out of line to compile once per
# permutation rather than once per codomain/domain split of it.
@noinline function permutedimsconjadd_flat!(
        dest,
        isconj::Bool,
        src,
        perm,
        α::Number,
        β::Number
    )
    check_input(bipermutedimsconjadd!, dest, isconj, src, perm, ())
    if SV.isstrided(dest) && SV.isstrided(src)
        permutedimsconjadd_strided!(
            SV.StridedView(dest), isconj, SV.StridedView(src), perm, α, β
        )
    else
        permutedimsconjadd_generic!(dest, isconj, src, perm, α, β)
    end
    return dest
end

permuteddims_perm(::PermutedDimsArray{<:Any, <:Any, perm}) where {perm} = perm
permuteddims_perm(a::PermutedDims) = a.perm

# `x -> α * conj(x)` or `x -> α * x`, with the choice made at run time so one compiled kernel
# serves both.
struct ConjScale{T}
    isconj::Bool
    α::T
end
(f::ConjScale)(x) = f.α * (f.isconj ? conj(x) : x)

# Its argument types depend only on the element types and the rank, so it compiles once per
# element type and rank.
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
        # Dispatched at run time, so the accumulating kernel only compiles when it is used.
        Base.inferencebarrier(permutedimsconjaccumulate_strided!)(
            dest, f, src_perm, convert(T, β)
        )
    end
    return dest
end
function permutedimsconjaccumulate_strided!(dest, f, src_perm, β)
    Strided._mapreducedim!(f, +, x -> β * x, size(dest), (dest, src_perm))
    return nothing
end

function permutedimsconjadd_generic!(dest, isconj::Bool, src, perm, α::Number, β::Number)
    src_perm = permuteddims(src, perm)
    f = ConjScale(isconj, α)
    if iszero(β)
        dest .= f.(src_perm)
    else
        # Dispatched at run time, so the accumulating kernel only compiles when it is used.
        Base.inferencebarrier(permutedimsconjaccumulate_generic!)(dest, f, src_perm, β)
    end
    return dest
end
function permutedimsconjaccumulate_generic!(dest, f, src_perm, β)
    dest .= β .* dest .+ f.(src_perm)
    return nothing
end

"""
    bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(op.(src), (perm_codomain..., perm_domain...))`, where `op`
is `identity` or `conj`.

See also [`bipermutedimsconjadd!`](@ref).
"""
function bipermutedimsopadd!(
        dest, op, src, perm_codomain, perm_domain, α::Number, β::Number
    )
    return bipermutedimsconjadd!(dest, _isconj(op), src, perm_codomain, perm_domain, α, β)
end

function _isconj(op)
    if op === conj
        return true
    elseif op === identity
        return false
    else
        throw(ArgumentError("`op` must be `identity` or `conj`, got `$op`"))
    end
end

"""
    bipermutedimsadd!(dest, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(src, (perm_codomain..., perm_domain...))`.

See also [`bipermutedimsconjadd!`](@ref).
"""
function bipermutedimsadd!(dest, src, perm_codomain, perm_domain, α::Number, β::Number)
    return bipermutedimsconjadd!(dest, false, src, perm_codomain, perm_domain, α, β)
end

# ---------------------------------------------------------------------------- #
# permutedims*add! — flat or split permutation
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β) -> dest
    permutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(op.(src), perm)`, where `op` is `identity` or `conj`. The
permutation is either flat or split into `perm_codomain` and `perm_domain`, as in
[`bipermutedimsopadd!`](@ref).
"""
function permutedimsopadd!(dest, op, src, perm, α::Number, β::Number)
    return bipermutedimsopadd!(dest, op, src, perm, (), α, β)
end
function permutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α::Number, β::Number)
    return bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β)
end

"""
    permutedimsconjadd!(dest, isconj::Bool, src, perm, α, β) -> dest
    permutedimsconjadd!(dest, isconj::Bool, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(src, perm)`, with `src` conjugated when `isconj` is `true`.
The permutation is either flat or split into `perm_codomain` and `perm_domain`, as in
[`bipermutedimsconjadd!`](@ref).
"""
function permutedimsconjadd!(dest, isconj::Bool, src, perm, α::Number, β::Number)
    return bipermutedimsconjadd!(dest, isconj, src, perm, (), α, β)
end
function permutedimsconjadd!(
        dest, isconj::Bool, src, perm_codomain, perm_domain, α::Number, β::Number
    )
    return bipermutedimsconjadd!(dest, isconj, src, perm_codomain, perm_domain, α, β)
end

"""
    permutedimsadd!(dest, src, perm, α, β) -> dest
    permutedimsadd!(dest, src, perm_codomain, perm_domain, α, β) -> dest

`dest = β * dest + α * permutedims(src, perm)`. The permutation is either flat or split into
`perm_codomain` and `perm_domain`, as in [`bipermutedimsadd!`](@ref).
"""
function permutedimsadd!(dest, src, perm, α::Number, β::Number)
    return bipermutedimsadd!(dest, src, perm, (), α, β)
end
function permutedimsadd!(dest, src, perm_codomain, perm_domain, α::Number, β::Number)
    return bipermutedimsadd!(dest, src, perm_codomain, perm_domain, α, β)
end

# ---------------------------------------------------------------------------- #
# Convenience functions that lower to permutedimsconjadd!
# ---------------------------------------------------------------------------- #

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

In-place counterpart of [`permutedims`](@ref): write the permuted `a` into `dest`. The
permutation is either flat or split into `perm_codomain` and `perm_domain`, as in
[`bipermutedims!`](@ref).
"""
permutedims!(dest, a, perm) = bipermutedims!(dest, a, perm, ())
function permutedims!(dest, a, perm_codomain, perm_domain)
    return bipermutedims!(dest, a, perm_codomain, perm_domain)
end

"""
    permutedims(a, perm)
    permutedims(a, perm_codomain, perm_domain)

Out-of-place permutation of `a`. A flat `perm` gives an all-codomain result, and a permutation
split into `perm_codomain` and `perm_domain` also splits the result, as in
[`bipermutedims`](@ref). A dense operand ignores the split and stores the result flat.
"""
permutedims(a, perm) = bipermutedims(a, perm, ())
permutedims(a, perm_codomain, perm_domain) = bipermutedims(a, perm_codomain, perm_domain)
