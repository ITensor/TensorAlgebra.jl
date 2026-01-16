using FunctionImplementations: permuteddims
using Strided: StridedView, isstrided

maybestrided(as::AbstractArray...) = all(isstrided, as) ? StridedView.(as) : as

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
    return _add!(maybestrided(dest, src)..., α, β)
end

function _add!(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    if iszero(β)
        dest .= α .* src
    else
        dest .= β .* dest .+ α .* src
    end
    return dest
end

"""
    permutedimsadd!(dest, src, perm, α, β)

`dest = β * dest + α * permutedims(src, perm)`.
"""
function permutedimsadd!(
        dest::AbstractArray, src::AbstractArray, perm, α::Number, β::Number
    )
    dest′, src′ = maybestrided(dest, src)
    permutedimsadd!_view(dest′, src′, perm, α, β)
    return dest′
end

function permutedimsadd!_view(
        dest::AbstractArray, src::AbstractArray, perm, α::Number, β::Number
    )
    src_permuted = permuteddims(src, perm)
    add!(dest, src_permuted, α, β)
    return dest
end
