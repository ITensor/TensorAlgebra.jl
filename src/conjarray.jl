"""
    ConjArray(a::AbstractArray)

Lazy conjugate of `a`, constructed by [`conjed`](@ref). Its axes are the conjugated (dualized)
axes of `a` and its elements are the conjugated elements of `a`. Permuting, matricizing or
contracting it does not materialize the conjugate. Scalar indexing is disallowed.
"""
struct ConjArray{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::P
end

"""
    conjed(a::AbstractArray)

Lazy conjugate of `a`, returning a [`ConjArray`](@ref). `conjed(conjed(a))` returns `a`.
"""
conjed(a::AbstractArray) = ConjArray(a)
conjed(a::ConjArray) = parent(a)

# Methods that assume `conj` acts elementwise on the stored data (`fill!`, `==`, `mapreduce`, `dot`
# with one lazy operand) are defined only for a dense parent. For a graded array, conjugation also
# dualizes the axes and can carry fermion signs.
const ConjDenseArray{T, N} = ConjArray{T, N, <:DenseArray{T, N}}

Base.parent(a::ConjArray) = a.parent

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

function Base.similar(a::ConjArray, ::Type{T}, dims::Dims) where {T}
    return similar(parent(a), T, dims)
end

# Materialize as the eager `conj.(a)`, so array types with their own `conj` (graded fermion signs)
# agree.
Base.copy(a::ConjArray) = copy(ConjBroadcasted(parent(a)))
Base.Array{T, N}(a::ConjArray{<:Any, N}) where {T, N} = convert(Array{T, N}, copy(a))
# Base `collect` iterates elements, which scalar indexing disallows.
Base.collect(a::ConjArray) = Array(a)
Base.conj(a::ConjArray) = parent(a)

function Base.copyto!(dest::AbstractArray, src::ConjArray)
    return permutedimsconjadd!(
        dest,
        true,
        parent(src),
        ntuple(identity, ndims(src)),
        true,
        false
    )
end
function Base.copyto!(dest::ConjArray, src::AbstractArray)
    permutedimsconjadd!(parent(dest), true, src, ntuple(identity, ndims(src)), true, false)
    return dest
end
function Base.copyto!(dest::ConjArray, src::ConjArray)
    copyto!(parent(dest), parent(src))
    return dest
end
function Base.fill!(a::ConjDenseArray, value)
    fill!(parent(a), conj(value))
    return a
end

Base.:(==)(a::ConjArray, b::ConjArray) = parent(a) == parent(b)
function Base.:(==)(a::ConjDenseArray, b::DenseArray)
    axes(a) == axes(b) || return false
    return mapreduce((x, y) -> conj(x) == y, &, parent(a), b; init = true)
end
Base.:(==)(a::DenseArray, b::ConjDenseArray) = b == a
LinearAlgebra.norm(a::ConjArray, p::Real = 2) = LinearAlgebra.norm(parent(a), p)
Base.iszero(a::ConjArray) = iszero(parent(a))
function LinearAlgebra.dot(a::ConjArray, b::ConjArray)
    return conj(LinearAlgebra.dot(parent(a), parent(b)))
end
LinearAlgebra.dot(a::ConjDenseArray, b::DenseArray) = transpose(vec(parent(a))) * vec(b)
LinearAlgebra.dot(a::DenseArray, b::ConjDenseArray) = conj(LinearAlgebra.dot(b, a))
function Base.mapreduce(f, op, a::ConjDenseArray; kwargs...)
    return mapreduce(f ∘ conj, op, parent(a); kwargs...)
end
Base.permutedims(a::ConjArray, perm) = permutedims(a, perm)

Base.show(io::IO, a::ConjArray) = print(io, "conjed(", parent(a), ")")
function Base.show(io::IO, mime::MIME"text/plain", a::ConjArray)
    print(io, "conjed of ")
    return show(io, mime, copy(a))
end

Base.Broadcast.broadcastable(a::ConjArray) = Base.Broadcast.broadcasted(conj, parent(a))
function Base.Broadcast.BroadcastStyle(::Type{<:ConjArray{<:Any, <:Any, P}}) where {P}
    return Base.Broadcast.BroadcastStyle(P)
end

linearbroadcasted(::typeof(conj), a::ConjArray) = parent(a)

_unwrap_conj(isconj::Bool, a) = (isconj, a)
_unwrap_conj(isconj::Bool, a::ConjArray) = (!isconj, parent(a))

function _bipermutedimsconjadd!(
        dest, isconj::Bool, src::ConjArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    return bipermutedimsconjadd!(
        dest, !isconj, parent(src), perm_codomain, perm_domain, α, β
    )
end
function _bipermutedimsconj(isconj::Bool, src::ConjArray, perm_codomain, perm_domain)
    return bipermutedimsconj(!isconj, parent(src), perm_codomain, perm_domain)
end

function _matricizeconj(isconj::Bool, a::ConjArray, perm_codomain, perm_domain)
    return matricizeconj(!isconj, parent(a), perm_codomain, perm_domain)
end

function _matricizeconjcopy(isconj::Bool, a::ConjArray, perm_codomain, perm_domain)
    return matricizeconjcopy(!isconj, parent(a), perm_codomain, perm_domain)
end
