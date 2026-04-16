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
    permutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β)

`dest = β * dest + α * permutedims(op.(src), (perm_codomain..., perm_domain...))`.

This is the primary overload point for downstream array types that want to
implement op-aware bipartitioned permutation + accumulation (e.g., fuse `conj`
into the copy, or use lazy wrappers like `StridedView` with op metadata).

The `op` is an element-wise linear map (e.g., `identity`, `conj`).

The default implementation flattens the bipartitioned permutation, applies `op`
element-wise, permutes, then accumulates via broadcasting with Strided.jl
optimization when possible.
"""
function permutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)

    # TODO: Remove this 0-dimensional special case once GradedArray is its own type
    # (not an alias for BlockSparseArray), so the GradedArray permutedimsopadd! overload
    # catches the 0-dimensional contraction result.
    if iszero(ndims(dest))
        dest[] = β * dest[] + α * op(src[])
        return dest
    end

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

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op.(src), perm)`.

Flat-permutation convenience overload. Forwards to the bipartitioned version
with `perm_domain = ()`.
"""
function permutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, op, src, perm, (), α, β)
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
