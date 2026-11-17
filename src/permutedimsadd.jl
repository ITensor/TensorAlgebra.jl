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

Check that `args` are valid arguments to `f`, throwing an error if they are not. Methods are
defined for [`bipermutedimsconjadd!`](@ref), [`contractpermconjadd!`](@ref),
[`unmatricize`](@ref), [`dotperm`](@ref) and [`dot`](@ref). An array type's own method of
one of these functions can call `check_input` to validate its arguments the same way the
generic method does.
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
conjugated when `isconj` is `true`. `(perm_codomain, perm_domain)` splits the dimensions of
`dest` into a codomain and a domain.

This is the function array types overload to customize the operation. Overloading
[`bipermutedimsopadd!`](@ref) has no effect.

See also [`bipermutedimsopadd!`](@ref).
"""
function bipermutedimsconjadd!(
        dest, isconj::Bool, src,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    # Lazy wrappers are unwrapped here rather than in methods dispatching on them, so they never
    # make an array type's own method ambiguous.
    if src isa Union{ConjArray, ConjBroadcasted}
        return bipermutedimsconjadd!(
            dest, !isconj, parent(src), perm_codomain, perm_domain, α, β
        )
    elseif src isa Union{PermutedDimsArray, PermutedDims}
        w = permuteddims_perm(src)
        return bipermutedimsconjadd!(
            dest, isconj, parent(src),
            map(j -> w[j], perm_codomain), map(j -> w[j], perm_domain), α, β
        )
    end
    check_input(bipermutedimsconjadd!, dest, isconj, src, perm_codomain, perm_domain)
    perm = (perm_codomain..., perm_domain...)
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
# element type and rank (see `precompile.jl`).
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
is `identity` or `conj`. `(perm_codomain, perm_domain)` splits the dimensions of `dest` into a
codomain and a domain.

See also [`bipermutedimsconjadd!`](@ref), the form array types overload.
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

# ---------------------------------------------------------------------------- #
# permutedimsopadd! — flat-permutation interface
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β) -> dest

`dest = β * dest + α * permutedims(op.(src), perm)`, where `op` is `identity` or `conj`.

See also [`permutedimsconjadd!`](@ref), the form array types overload.
"""
function permutedimsopadd!(dest, op, src, perm, α::Number, β::Number)
    return permutedimsconjadd!(dest, _isconj(op), src, perm, α, β)
end
"""
    permutedimsconjadd!(dest, isconj::Bool, src, perm, α, β) -> dest

`dest = β * dest + α * permutedims(src, perm)`, with `src` conjugated when `isconj` is `true`.

This is the function array types overload to customize the operation, or they overload
[`bipermutedimsconjadd!`](@ref), which it calls. Overloading [`permutedimsopadd!`](@ref) has
no effect.

See also [`permutedimsopadd!`](@ref).
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

Out-of-place permutation of `a`. The single-permutation
form reorders every dimension into `perm`, giving an all-codomain result. The
two-permutation form additionally splits the dimensions into a codomain/domain
bipartition, with `perm_codomain` selecting the codomain dimensions and `perm_domain`
the domain ones.

A dense operand ignores the bipartition and stores the result flat.
"""
function permutedims(a, perm)
    dest = similar_map(a, eltype(a), map(p -> axes(a, p), perm), ())
    return permutedims!(dest, a, perm)
end
function permutedims(a, perm_codomain, perm_domain)
    axes_codomain = map(p -> axes(a, p), perm_codomain)
    # `similar_map` dualizes the domain axes it is given, so pass them pre-conjugated.
    axes_domain = map(p -> conj(axes(a, p)), perm_domain)
    dest = similar_map(a, eltype(a), axes_codomain, axes_domain)
    return permutedims!(dest, a, perm_codomain, perm_domain)
end
