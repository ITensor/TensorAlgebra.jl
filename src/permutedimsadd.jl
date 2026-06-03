using FunctionImplementations: permuteddims
using Strided: Strided
using StridedViews: StridedViews as SV

# Specify if an array is on CPU. This is helpful for backends that don't support
# operations on GPU, such as Strided.jl.
iscpu(::AbstractArray) = true
# Convert to StridedView only if all arrays are strided and on CPU.
function maybestrided(as::AbstractArray...)
    return all(a -> SV.isstrided(a) && iscpu(a), as) ? SV.StridedView.(as) : as
end

# ---------------------------------------------------------------------------- #
# bipermutedimsopadd! — the primary materialization primitive
# ---------------------------------------------------------------------------- #

function bipermutedimsopadd! end

function check_input(
        ::typeof(bipermutedimsopadd!), dest::AbstractArray, src::AbstractArray,
        perm_codomain, perm_domain
    )
    perm = (perm_codomain..., perm_domain...)
    ndims(dest) == length(perm) ||
        throw(DimensionMismatch("destination ndims does not match permutation length"))
    axes(dest) == ntuple(d -> axes(src, perm[d]), ndims(dest)) ||
        throw(DimensionMismatch("destination axes do not match permuted source axes"))
    return nothing
end

"""
    bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, α, β)

`dest = β * dest + α * permutedims(op.(src), (perm_codomain..., perm_domain...))`.

This is the primary overload point for downstream array types that want to
implement op-aware bipartitioned permutation + accumulation (e.g., fuse `conj`
into the copy, or use lazy wrappers like `StridedView` with op metadata).

The `op` is an element-wise linear map (e.g., `identity`, `conj`).

The default implementation flattens the bipartitioned permutation, applies `op`
element-wise, permutes, then accumulates via broadcasting with Strided.jl
optimization when possible.
"""
function bipermutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)
    check_input(bipermutedimsopadd!, dest, src, perm_codomain, perm_domain)

    # 0-dim short-circuit: avoid the permute-broadcast path entirely so that
    # downstream array types (e.g. `BlockSparseArray{T, 0}`) don't have to define
    # `getindex` on a 0-dim `PermutedDimsArray` wrapper around them.
    # The `iszero(β)` guard follows the BLAS convention that `β = 0` means `dest`
    # is write-only — its slot need not be defined. This matters for element types
    # whose `undef` storage is unreadable, e.g. `Array{BigFloat, 0}(undef)[]` throws
    # `UndefRefError`.
    if iszero(ndims(dest))
        if iszero(β)
            dest[] = α * op(src[])
        else
            dest[] = β * dest[] + α * op(src[])
        end
        return dest
    end

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
# permutedimsopadd! — flat-permutation interface
# ---------------------------------------------------------------------------- #

"""
    permutedimsopadd!(dest, op, src, perm, α, β)

`dest = β * dest + α * permutedims(op.(src), perm)`.

This is the single materialization primitive for `LinearBroadcasted` types.
Downstream array types should implement `bipermutedimsopadd!` for the
bipartitioned permutation version; this flat-permutation overload forwards to it
with `perm_domain = ()`.
"""
function permutedimsopadd!(
        dest::AbstractArray, op, src::AbstractArray, perm, α::Number, β::Number
    )
    return bipermutedimsopadd!(dest, op, src, perm, (), α, β)
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
