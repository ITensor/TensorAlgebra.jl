"""
    ConjArray(a::AbstractArray)

Lazy conjugate of `a`: an `AbstractArray` whose axes are the conjugated (dualized) parent axes
and whose elements are the conjugated parent elements. Constructed by [`conjed`](@ref).
Permutation, matricization and contraction absorb it by unwrapping to the parent and folding
`conj` into their `op`, so it is never materialized on its way into those operations. Whole-array
reads, writes, broadcasting and reductions go through the parent as well, and scalar indexing is
disallowed, so an operation without such a path errors instead of silently looping over elements.
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

function Base.getindex(a::ConjArray, I::Int...)
    return throw(
        ArgumentError(
            "scalar indexing of a lazy `ConjArray` is disallowed, materialize it with `copy` first"
        )
    )
end
function Base.setindex!(a::ConjArray, value, I::Int...)
    return throw(
        ArgumentError(
            "scalar indexing of a lazy `ConjArray` is disallowed, write the whole array with `copyto!`"
        )
    )
end

# Allocate like the parent, so a graded parent yields a graded array.
function Base.similar(a::ConjArray, ::Type{T}, dims::Dims) where {T}
    return similar(parent(a), T, dims)
end

# Materialize through the eager `conj.(a)` path, so a lazy and an eager conjugate agree
# (including any parent-specific behavior such as a graded fermion sign).
Base.copy(a::ConjArray) = copy(ConjBroadcasted(parent(a)))
Base.Array{T, N}(a::ConjArray{<:Any, N}) where {T, N} = convert(Array{T, N}, copy(a))
# Base `collect` iterates elements, which scalar indexing disallows.
Base.collect(a::ConjArray) = Array(a)
# Like `conj(::Adjoint)` in LinearAlgebra, this returns the parent rather than a copy.
Base.conj(a::ConjArray) = parent(a)

function Base.copyto!(dest::AbstractArray, src::ConjArray)
    return permutedimsopadd!(
        dest,
        conj,
        parent(src),
        ntuple(identity, ndims(src)),
        true,
        false
    )
end
# Write into the parent the conjugate of what the lazy conjugate should hold. `conj` is an
# involution, fermion signs included, so this is exact.
function Base.copyto!(dest::ConjArray, src::AbstractArray)
    permutedimsopadd!(parent(dest), conj, src, ntuple(identity, ndims(src)), true, false)
    return dest
end
# Also resolves the ambiguity between the two methods above.
function Base.copyto!(dest::ConjArray, src::ConjArray)
    copyto!(parent(dest), parent(src))
    return dest
end
function Base.fill!(a::ConjArray, value)
    fill!(parent(a), conj(value))
    return a
end

# Also resolves the ambiguity between the two mixed methods below.
Base.:(==)(a::ConjArray, b::ConjArray) = parent(a) == parent(b)
# A fused two-array reduction rather than a copy, so it maps to a single kernel on GPU.
function Base.:(==)(a::ConjArray, b::AbstractArray)
    axes(a) == axes(b) || return false
    return mapreduce((x, y) -> conj(x) == y, &, parent(a), b; init = true)
end
Base.:(==)(a::AbstractArray, b::ConjArray) = b == a
# Conjugation, and any sign it carries, preserves every p-norm. `isapprox` goes through `norm`.
LinearAlgebra.norm(a::ConjArray, p::Real = 2) = LinearAlgebra.norm(parent(a), p)
Base.iszero(a::ConjArray) = iszero(parent(a))
# `conj` is antiunitary, fermion signs included, so a pair of lazy conjugates reduces to the parents.
# This also resolves the ambiguity between the two mixed methods below.
function LinearAlgebra.dot(a::ConjArray, b::ConjArray)
    return conj(LinearAlgebra.dot(parent(a), parent(b)))
end
# Against a plain array, the lazy conjugate cancels the conjugation in `dot`, leaving the bilinear
# pairing of the parent with `b` (BLAS `dotu` for dense arrays).
LinearAlgebra.dot(a::ConjArray, b::AbstractArray) = transpose(vec(parent(a))) * vec(b)
LinearAlgebra.dot(a::AbstractArray, b::ConjArray) = conj(LinearAlgebra.dot(b, a))
# Resolve ambiguities with LinearAlgebra's `dot(::AbstractMatrix, ::Diagonal)` and its reverse.
function LinearAlgebra.dot(a::ConjArray{<:Any, 2}, b::LinearAlgebra.Diagonal)
    return transpose(vec(parent(a))) * vec(b)
end
function LinearAlgebra.dot(a::LinearAlgebra.Diagonal, b::ConjArray{<:Any, 2})
    return conj(LinearAlgebra.dot(b, a))
end
function Base.mapreduce(f, op, a::ConjArray; kwargs...)
    return mapreduce(f ∘ conj, op, parent(a); kwargs...)
end
Base.permutedims(a::ConjArray, perm) = permutedims(a, perm)

Base.show(io::IO, a::ConjArray) = print(io, "conjed(", parent(a), ")")
function Base.show(io::IO, mime::MIME"text/plain", a::ConjArray)
    print(io, "conjed of ")
    return show(io, mime, copy(a))
end

# A broadcast over a lazy conjugate reads it as `conj.(parent)`, so it never indexes the wrapper
# and a linear one lowers to a `ConjBroadcasted` leaf.
Base.Broadcast.broadcastable(a::ConjArray) = Base.Broadcast.broadcasted(conj, parent(a))
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
function matricizeopcopy(op, a::ConjArray, perm_codomain, perm_domain)
    return matricizeopcopy(_compose_op(op, conj), parent(a), perm_codomain, perm_domain)
end

_unwrap_conj(op, a) = (op, a)
_unwrap_conj(op, a::ConjArray) = (_compose_op(op, conj), parent(a))
