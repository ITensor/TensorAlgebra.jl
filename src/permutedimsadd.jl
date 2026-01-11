using FunctionImplementations: permuteddims
using Strided: StridedView, isstrided

maybestrided(as::AbstractArray...) = all(isstrided, as) ? StridedView.(as) : as

"""
    add!(dest, src, α, β)

`dest = β * dest + α * src`.
"""
function add!(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
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
