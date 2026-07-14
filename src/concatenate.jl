# `concatenate`/`concatenate!` over arrays, with the destination chosen from all inputs rather than
# just the first (unlike `Base.cat`). Backends customize by overloading `cat_similar` (destination)
# and `cat_copyto!` (placement) on the combined `cat_style` of the arguments.

import Base.Broadcast as BC
using Base: promote_eltypeof

function cat_axis(
        a1::AbstractUnitRange, a2::AbstractUnitRange, a_rest::AbstractUnitRange...
    )
    return cat_axis(cat_axis(a1, a2), a_rest...)
end
function cat_axis(a1::AbstractUnitRange, a2::AbstractUnitRange)
    first(a1) == first(a2) == 1 || throw(ArgumentError("Concatenated axes must start at 1"))
    return Base.OneTo(length(a1) + length(a2))
end

cat_ndims(dims, as::AbstractArray...) = cat_ndims(Val(dims), as...)
function cat_ndims(dims::Val, as::AbstractArray...)
    return max(maximum(unval(dims)), maximum(ndims, as))
end

cat_axes(dims, as::AbstractArray...) = cat_axes(Val(dims), as...)
function cat_axes(dims::Val, a::AbstractArray, as::AbstractArray...)
    return ntuple(cat_ndims(dims, a, as...)) do dim
        return if dim in unval(dims)
            cat_axis(map(Base.Fix2(axes, dim), (a, as...))...)
        else
            axes(a, dim)
        end
    end
end

function cat_style(dims, as::AbstractArray...)
    N = cat_ndims(dims, as...)
    return typeof(BC.combine_styles(as...))(Val(N))
end

# Default destination: reuse broadcast's `similar` for the style by wrapping in a `Broadcasted`.
function cat_similar(style, ::Type{T}, ax, args...) where {T}
    return similar(BC.Broadcasted(style, identity, args, ax), T)
end

concatenate(dims, args...) = concatenate(Val(dims), args...)
function concatenate(dims::Val, args...)
    style = cat_style(dims, args...)
    dest = cat_similar(style, promote_eltypeof(args...), cat_axes(dims, args...), args...)
    return cat_copyto!(dest, style, dims, args...)
end

function concatenate!(dest, args...; dims)
    return cat_copyto!(dest, cat_style(dims, args...), dims, args...)
end

# The offset placement below is adapted from Base's `cat`:
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

dims2cat(dims) = dims2cat(Val(dims))
function dims2cat(dims::Val)
    d = unval(dims)
    if any(≤(0), d)
        throw(ArgumentError("All cat dimensions must be positive integers, but got $d"))
    end
    return ntuple(in(d), maximum(d))
end

# The default strips the style to `nothing`, so a backend can instead specialize on `typeof(dest)`
# without ambiguity against the style dispatch.
cat_copyto!(dest, style, dims, args...) = cat_copyto!(dest, nothing, dims, args...)
function cat_copyto!(dest, ::Nothing, dims, args...)
    catdims = dims2cat(dims)
    shape = map(length, cat_axes(dims, args...))
    count(!iszero, catdims)::Int > 1 && zero!(dest)
    return __cat!(dest, shape, catdims, args...)
end
