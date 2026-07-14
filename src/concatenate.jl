# Concatenation interface, vendored flat from the `Concatenate` module that previously lived in
# SparseArraysBase / FunctionImplementations. TensorAlgebra owns it here as the single source of
# truth: `cat`/`cat!`/`concatenate`, plus the backend hooks `cat_similar` (destination allocation)
# and `cat_copyto!` (placement). Both dispatch on the combined `cat_style` of the arguments, so a
# backend (SparseArraysBase, GradedArrays, a TensorKit ext) customizes concatenation by overloading
# `cat_similar`/`cat_copyto!` on its style, without scalar indexing.
#
# This is a function interface (no lazy `Concatenated` wrapper): concatenation does not fuse, so
# there is nothing for a lazy object to defer. `cat`/`cat!` resolve the style with `cat_style`,
# allocate with `cat_similar`, and place with `cat_copyto!`. Mostly a copy of Base's `cat`, except
# the destination is chosen from all inputs instead of just the first.

import Base.Broadcast as BC
using Base: promote_eltypeof

_valdims(dims::Val) = dims
_valdims(dims) = Val(dims)

# Concatenation axes and style, computed directly from `dims` and the arguments.
# ------------------------------------------------------------------------------
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

# Allocate the destination container.
# -----------------------------------
# Allocate the destination for concatenating `args` along the combined `style`, with element type
# `T` and axes `ax`. Backends override on their style; the default reuses broadcast's `similar` by
# building a `Broadcasted` with the same style and axes.
function cat_similar(style, ::Type{T}, ax, args...) where {T}
    return similar(BC.Broadcasted(style, identity, args, ax), T)
end

# Main logic.
# -----------
# Concatenate the supplied `args` along dimensions `dims`.
concatenate(dims, args...) = concatenate(_valdims(dims), args...)
function concatenate(dims::Val, args...)
    style = cat_style(dims, args...)
    dest = cat_similar(style, promote_eltypeof(args...), cat_axes(dims, args...), args...)
    return cat_copyto!(dest, style, dims, args...)
end

# Concatenate the supplied `args` along dimensions `dims`.
cat(args...; dims) = concatenate(dims, args...)

# Concatenate the supplied `args` along dimensions `dims`, placing the result into `dest`.
function cat!(dest, args...; dims)
    d = _valdims(dims)
    return cat_copyto!(dest, cat_style(d, args...), d, args...)
end

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

# Materialize the concatenation into `dest`. Backends override on their `style`; the default strips
# the style to `nothing` (which lets a backend instead specialize on `typeof(dest)` without
# ambiguity against the style dispatch) and runs the generic offset placement.
cat_copyto!(dest, style, dims, args...) = cat_copyto!(dest, nothing, dims, args...)
function cat_copyto!(dest, ::Nothing, dims, args...)
    catdims = dims2cat(dims)
    shape = map(length, cat_axes(dims, args...))
    count(!iszero, catdims)::Int > 1 && zero!(dest)
    return __cat!(dest, shape, catdims, args...)
end
