# Lazy conjugated-array wrapper, mirroring `Adjoint`/`adjoint`. `conjed(a)` is the
# friendly lazy companion to the eager `conj(a)`. The op primitives absorb a `ConjArray`
# by unwrapping to the parent and folding `conj` into `op` (see `bipermutedimsopadd!`
# below), so conjugation flows through broadcasting, contraction, and matricize uniformly.

"""
    ConjArray(a::AbstractArray)

Lazy conjugate of `a`: an `AbstractArray` whose axes are the conjugated parent axes and
whose elements are the conjugated parent elements. Constructed by [`conjed`](@ref). The op
primitives absorb it by unwrapping to the parent and folding `conj` into their `op`.
"""
struct ConjArray{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::P
end

"""
    conjed(a::AbstractArray)

Lazy conjugate of `a`, returning a `ConjArray`. The lazy companion to the eager `conj(a)`,
mirroring `adjoint`/`Adjoint`. `conjed(conjed(a))` returns `a`.
"""
conjed(a::AbstractArray) = ConjArray(a)
conjed(a::ConjArray) = parent(a)

Base.parent(a::ConjArray) = a.parent

# `conj` of an axis: dualizes graded axes, identity for plain ranges (Base `conj` on a
# range broadcasts to a vector, so we cannot call it directly). Overridable downstream.
conjaxis(r) = r

Base.axes(a::ConjArray) = map(conjaxis, axes(parent(a)))
Base.size(a::ConjArray) = size(parent(a))

Base.IndexStyle(::Type{<:ConjArray{<:Any, <:Any, P}}) where {P} = IndexStyle(P)
Base.getindex(a::ConjArray, I::Int...) = conj(parent(a)[I...])

# Absorb a `ConjArray` source by unwrapping to the parent and folding `conj` into `op`
# (`identity -> conj`, `conj -> identity`). Mirrors the `PermutedDimsArray` absorption,
# which instead folds its permutation into `perm`.
function bipermutedimsopadd!(
        dest::AbstractArray, op, src::ConjArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    return bipermutedimsopadd!(
        dest, _compose_op(op, conj), parent(src),
        perm_codomain, perm_domain,
        α, β
    )
end

# Materialize eagerly through the op primitive, so the conjugation (and any parent-specific
# behavior such as a graded fermion sign) is applied by the same `op = conj` path the
# primitives use everywhere else. `copy(conjed(a))` for a plain `Array` is `conj(a)`.
Base.copy(a::ConjArray) = add!(similar(parent(a), eltype(a), axes(a)), a, true, false)
Base.Broadcast.materialize(a::ConjArray) = copy(a)

# Broadcast like the parent (e.g. preserve a graded style), not the default array style.
function Base.Broadcast.BroadcastStyle(::Type{<:ConjArray{<:Any, <:Any, P}}) where {P}
    return Base.Broadcast.BroadcastStyle(P)
end
