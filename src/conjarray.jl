"""
    ConjArray(a::AbstractArray)

Lazy conjugate of `a`: an `AbstractArray` whose axes are the conjugated (dualized) parent axes
and whose elements are the conjugated parent elements. Constructed by [`conjed`](@ref).
Permutation, matricization and contraction absorb it by unwrapping to the parent and folding
`conj` into their `op`, so it is never materialized on its way into those operations.
"""
struct ConjArray{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::P
end

"""
    conjed(a::AbstractArray)

Lazy conjugate of `a`, returning a [`ConjArray`](@ref). The lazy companion to the eager
`conj(a)`, mirroring `adjoint`/`Adjoint`. `conjed(conjed(a))` returns `a`.
"""
conjed(a::AbstractArray) = ConjArray(a)
conjed(a::ConjArray) = parent(a)

Base.parent(a::ConjArray) = a.parent

# Conjugating an axis is identity for plain integer ranges and dualizes graded axes (where
# `conj` is overloaded as `dual` downstream).
Base.axes(a::ConjArray) = map(conj, axes(parent(a)))
Base.size(a::ConjArray) = size(parent(a))

Base.IndexStyle(::Type{<:ConjArray{<:Any, <:Any, P}}) where {P} = IndexStyle(P)
Base.getindex(a::ConjArray, I::Int...) = conj(parent(a)[I...])

# Allocate like the parent, so a graded parent yields a graded array.
Base.similar(a::ConjArray, ::Type{T}, ax::Tuple) where {T} = similar(parent(a), T, ax)
function similar_map(a::ConjArray, ::Type{T}, axes_codomain, axes_domain) where {T}
    return similar_map(parent(a), T, axes_codomain, axes_domain)
end

# Materialize through the eager `conj.(a)` path, so a lazy and an eager conjugate agree
# (including any parent-specific behavior such as a graded fermion sign).
Base.copy(a::ConjArray) = copy(ConjBroadcasted(parent(a)))
# Like `conj(::Adjoint)` in LinearAlgebra, this returns the parent rather than a copy.
Base.conj(a::ConjArray) = parent(a)

function Base.Broadcast.BroadcastStyle(::Type{<:ConjArray{<:Any, <:Any, P}}) where {P}
    return Base.Broadcast.BroadcastStyle(P)
end

linearbroadcasted(::typeof(conj), a::ConjArray) = parent(a)

# Each operation unwraps a `ConjArray` to its parent and folds `conj` into `op`
# (`identity -> conj`, `conj -> identity`), before any backend dispatch.
function bipermutedimsopadd!(
        dest, op, src::ConjArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    return bipermutedimsopadd!(
        dest, _compose_op(op, conj), parent(src),
        perm_codomain, perm_domain,
        α, β
    )
end
function permutedimsop(op, src::ConjArray, perm_codomain, perm_domain)
    return permutedimsop(_compose_op(op, conj), parent(src), perm_codomain, perm_domain)
end
function matricizeop(op, a::ConjArray, perm_codomain, perm_domain)
    return matricizeop(_compose_op(op, conj), parent(a), perm_codomain, perm_domain)
end

_unwrap_conj(op, a) = (op, a)
_unwrap_conj(op, a::ConjArray) = (_compose_op(op, conj), parent(a))
