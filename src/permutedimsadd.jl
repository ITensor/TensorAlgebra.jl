import StridedViews as SV
using FunctionImplementations: permuteddims

# Specify if an array is on CPU. This is helpful for backends that don't support
# operations on GPU, such as Strided.jl.
iscpu(::AbstractArray) = true
# Convert to StridedView only if all arrays are strided and on CPU.
function maybestrided(as::AbstractArray...)
    return all(a -> SV.isstrided(a) && iscpu(a), as) ? SV.StridedView.(as) : as
end

"""
    add!(dest, src)

Equivalent to `dest .+= src`, but maybe with a more optimized/specialized implementation.
Generally calls `add!(dest, src, true, true)`.
"""
add!(dest::AbstractArray, src::AbstractArray) = add!(dest, src, true, true)

"""
    add!(dest, src, α, β)

Equivalent to `dest .= β .* dest .+ α .* src`, but maybe with a more optimized/specialized
implementation.
"""
function add!(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    add!_broadcast(maybestrided(dest, src)..., α, β)
    return dest
end

# Broadcasting implementation of add!.
function add!_broadcast(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    # This works around a bug in Strided.jl v2.3.4 and below when broadcasting
    # empty StridedViews: https://github.com/QuantumKitHub/Strided.jl/pull/50
    # TODO: Delete this and bump the version of Strided.jl once that is fixed.
    isempty(dest) && return dest

    if iszero(β)
        dest .= α .* src
    else
        dest .= β .* dest .+ α .* src
    end
    return dest
end

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op(src), perm)`.

The `op` is an element-wise linear map applied to `src` before permutation and
accumulation. Downstream array types can specialize on specific `op`s (e.g.,
`op::typeof(conj)`) to fuse the operation without allocating.

The default implementation eagerly applies `op`: `permutedimsadd!(dest, op.(src), perm, α, β)`.
"""
function permutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray, perm, α::Number, β::Number
    )
    return permutedimsadd!(dest, op.(src), perm, α, β)
end

"""
    permutedimsadd!(dest, src, perm, α, β)

`dest = β * dest + α * permutedims(src, perm)`.
"""
function permutedimsadd!(
        dest::AbstractArray, src::AbstractArray, perm, α::Number, β::Number
    )
    return add!(dest, permuteddims(src, perm), α, β)
end
