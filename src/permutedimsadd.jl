import StridedViews as SV
using FunctionImplementations: permuteddims

# Specify if an array is on CPU. This is helpful for backends that don't support
# operations on GPU, such as Strided.jl.
iscpu(::AbstractArray) = true
# Convert to StridedView only if all arrays are strided and on CPU.
function maybestrided(as::AbstractArray...)
    return all(a -> SV.isstrided(a) && iscpu(a), as) ? SV.StridedView.(as) : as
end

# ---------------------------------------------------------------------------- #
# permutedimsopadd! — the single materialization primitive
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op.(src), perm)`.

This is the single materialization primitive for `LinearBroadcasted` types.
Downstream array types should implement this function. The `op` is an element-wise
linear map (e.g., `identity`, `conj`, `adjoint`, `transpose`, `Float32`).

The default implementation applies `op` element-wise, permutes, then accumulates
via broadcasting with Strided.jl optimization when possible.
"""
function permutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray, perm, α::Number, β::Number
    )
    # This works around a bug in Strided.jl v2.3.4 and below when broadcasting
    # empty StridedViews: https://github.com/QuantumKitHub/Strided.jl/pull/50
    # TODO: Delete this and bump the version of Strided.jl once that is fixed.
    isempty(dest) && return dest

    dest′, src′ = maybestrided(dest, permuteddims(src, perm))
    if op === identity
        if iszero(β)
            dest′ .= α .* src′
        else
            dest′ .= β .* dest′ .+ α .* src′
        end
    else
        if iszero(β)
            dest′ .= α .* op.(src′)
        else
            dest′ .= β .* dest′ .+ α .* op.(src′)
        end
    end
    return dest
end

# ---------------------------------------------------------------------------- #
# Convenience functions that lower to permutedimsopadd!
# ---------------------------------------------------------------------------- #

"""
    permutedimsadd!(dest, src, perm, α, β)

`dest = β * dest + α * permutedims(src, perm)`.
"""
function permutedimsadd!(
        dest::AbstractArray, src::AbstractArray, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, identity, src, perm, α, β)
end

"""
    add!(dest, src, α, β)

`dest = β * dest + α * src`.
"""
function add!(dest::AbstractArray, src, α::Number, β::Number)
    return permutedimsopadd!(dest, identity, src, ntuple(identity, ndims(src)), α, β)
end

"""
    add!(dest, src)

`dest .+= src`.
"""
add!(dest::AbstractArray, src) = add!(dest, src, true, true)
