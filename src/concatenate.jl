# Concatenation interface, vendored flat from the `Concatenate` module that previously lived in
# SparseArraysBase / FunctionImplementations (both carried identical copies). TensorAlgebra owns it
# here as the single source of truth: `cat`/`cat!`/`concatenate` and the lazy `Concatenated` object,
# with a generic `copyto!(::Concatenated{Nothing})` fallback. Backends (SparseArraysBase,
# GradedArrays, a TensorKit ext) specialize `copyto!(dest, ::Concatenated{<:TheirStyle})` for
# block-level placement without scalar indexing.
#
# Alternative implementation for `Base.cat` through `cat`/`cat!`. This is mostly a copy of the Base
# implementation, with the main difference being that the destination is chosen based on all inputs
# instead of just the first. There is an intermediate representation in terms of a `Concatenated`
# object, reminiscent of how Broadcast works. Destination selection can be customized through
# `Base.similar(::Concatenated{Style}, ::Type{T}, axes)`, and the operation itself through
# `Base.copy`/`Base.copyto!` on a `Concatenated`.

import Base.Broadcast as BC
using Base: promote_eltypeof

function _Concatenated end

# Lazy representation of the concatenation of various `Args` along `Dims`, in order to
# provide hooks to customize the implementation.
struct Concatenated{Style, Dims, Args <: Tuple}
    style::Style
    dims::Val{Dims}
    args::Args
    global @inline function _Concatenated(
            style::Style, dims::Val{Dims}, args::Args
        ) where {Style, Dims, Args <: Tuple}
        return new{Style, Dims, Args}(style, dims, args)
    end
end

function Concatenated(
        style::Union{BC.AbstractArrayStyle, Nothing}, dims::Val, args::Tuple
    )
    return _Concatenated(style, dims, args)
end
function Concatenated(dims::Val, args::Tuple)
    return Concatenated(cat_style(dims, args...), dims, args)
end
function Concatenated{Style}(
        dims::Val, args::Tuple
    ) where {Style <: Union{BC.AbstractArrayStyle, Nothing}}
    return Concatenated(Style(), dims, args)
end

concatenated_dims(::Concatenated{<:Any, D}) where {D} = D
concatenated_style(concat::Concatenated) = getfield(concat, :style)

concatenated(dims, args...) = concatenated(Val(dims), args...)
concatenated(dims::Val, args...) = Concatenated(dims, args)

function Base.convert(
        ::Type{Concatenated{NewStyle}}, concat::Concatenated{<:Any, Dims, Args}
    ) where {NewStyle, Dims, Args}
    return Concatenated{NewStyle}(
        concat.dims, concat.args
    )::Concatenated{NewStyle, Dims, Args}
end

# allocating the destination container
# ------------------------------------
Base.similar(concat::Concatenated) = similar(concat, eltype(concat))
Base.similar(concat::Concatenated, ::Type{T}) where {T} = similar(concat, T, axes(concat))
function Base.similar(concat::Concatenated, ax)
    return similar(concat, eltype(concat), ax)
end

function Base.similar(concat::Concatenated, ::Type{T}, ax) where {T}
    # Convert to a broadcasted to leverage its similar implementation.
    bc = BC.Broadcasted(concatenated_style(concat), identity, concat.args, ax)
    return similar(bc, T)
end

function cat_axis(
        a1::AbstractUnitRange, a2::AbstractUnitRange, a_rest::AbstractUnitRange...
    )
    return cat_axis(cat_axis(a1, a2), a_rest...)
end
function cat_axis(a1::AbstractUnitRange, a2::AbstractUnitRange)
    first(a1) == first(a2) == 1 || throw(ArgumentError("Concatenated axes must start at 1"))
    return Base.OneTo(length(a1) + length(a2))
end

function cat_ndims(dims, as::AbstractArray...)
    return max(maximum(dims), maximum(ndims, as))
end
function cat_ndims(dims::Val, as::AbstractArray...)
    return cat_ndims(unval(dims), as...)
end

function cat_axes(dims, a::AbstractArray, as::AbstractArray...)
    return ntuple(cat_ndims(dims, a, as...)) do dim
        return if dim in dims
            cat_axis(map(Base.Fix2(axes, dim), (a, as...))...)
        else
            axes(a, dim)
        end
    end
end
function cat_axes(dims::Val, as::AbstractArray...)
    return cat_axes(unval(dims), as...)
end

function cat_style(dims, as::AbstractArray...)
    N = cat_ndims(dims, as...)
    return typeof(BC.combine_styles(as...))(Val(N))
end

Base.eltype(concat::Concatenated) = promote_eltypeof(concat.args...)
Base.axes(concat::Concatenated) = cat_axes(concatenated_dims(concat), concat.args...)
Base.size(concat::Concatenated) = length.(axes(concat))
Base.ndims(concat::Concatenated) = cat_ndims(concatenated_dims(concat), concat.args...)

# Main logic
# ----------
# Concatenate the supplied `args` along dimensions `dims`.
concatenate(dims, args...) = Base.materialize(concatenated(dims, args...))

# Concatenate the supplied `args` along dimensions `dims`.
cat(args...; dims) = concatenate(dims, args...)
Base.materialize(concat::Concatenated) = copy(concat)

# Concatenate the supplied `args` along dimensions `dims`, placing the result into `dest`.
function cat!(dest, args...; dims)
    Base.materialize!(dest, concatenated(dims, args...))
    return dest
end
Base.materialize!(dest, concat::Concatenated) = copyto!(dest, concat)

Base.copy(concat::Concatenated) = copyto!(similar(concat), concat)

# The following is largely copied from the Base implementation of `Base.cat`, see:
# https://github.com/JuliaLang/julia/blob/885b1cd875f101f227b345f681cc36879124d80d/base/abstractarray.jl#L1778-L1887
_copy_or_fill!(A, inds, x) = fill!(view(A, inds...), x)
_copy_or_fill!(A, inds, x::AbstractArray) = (A[inds...] = x)

cat_size(A) = (1,)
cat_size(A::AbstractArray) = size(A)
cat_size(A, d) = 1
cat_size(A::AbstractArray, d) = size(A, d)

cat_indices(A, d) = Base.OneTo(1)
cat_indices(A::AbstractArray, d) = axes(A, d)

function __cat!(A, shape, catdims, X...)
    return __cat_offset!(A, shape, catdims, ntuple(zero, length(shape)), X...)
end
function __cat_offset!(A, shape, catdims, offsets, x, X...)
    # splitting the "work" on x from X... may reduce latency (fewer costly specializations)
    newoffsets = __cat_offset1!(A, shape, catdims, offsets, x)
    return __cat_offset!(A, shape, catdims, newoffsets, X...)
end
__cat_offset!(A, shape, catdims, offsets) = A
function __cat_offset1!(A, shape, catdims, offsets, x)
    inds = ntuple(length(offsets)) do i
        return if (i <= length(catdims) && catdims[i])
            offsets[i] .+ cat_indices(x, i)
        else
            1:shape[i]
        end
    end
    _copy_or_fill!(A, inds, x)
    newoffsets = ntuple(length(offsets)) do i
        return if (i <= length(catdims) && catdims[i])
            offsets[i] + cat_size(x, i)
        else
            offsets[i]
        end
    end
    return newoffsets
end

dims2cat(dims::Val) = dims2cat(unval(dims))
function dims2cat(dims)
    if any(≤(0), dims)
        throw(ArgumentError("All cat dimensions must be positive integers, but got $dims"))
    end
    return ntuple(in(dims), maximum(dims))
end

# default falls back to replacing style with Nothing
# this permits specializing on typeof(dest) without ambiguities
# Note: this needs to be defined for AbstractArray specifically to avoid ambiguities with Base.
@inline function Base.copyto!(dest::AbstractArray, concat::Concatenated)
    return copyto!(dest, convert(Concatenated{Nothing}, concat))
end

function Base.copyto!(dest::AbstractArray, concat::Concatenated{Nothing})
    catdims = dims2cat(concatenated_dims(concat))
    shape = size(concat)
    count(!iszero, catdims)::Int > 1 && zero!(dest)
    return __cat!(dest, shape, catdims, concat.args...)
end
