# Workaround for https://github.com/JuliaLang/julia/issues/52615.
# Fixed by https://github.com/JuliaLang/julia/pull/52623.
# TODO remove once support for Julia 1.10 is dropped
function _permutedims!(
        a_dest::AbstractArray{<:Any, N}, a_src::AbstractArray{<:Any, N},
        perm::Tuple{Vararg{Int, N}}
    ) where {N}
    permutedims!(a_dest, a_src, perm)
    return a_dest
end
function _permutedims!(
        a_dest::AbstractArray{<:Any, 0}, a_src::AbstractArray{<:Any, 0}, perm::Tuple{}
    )
    a_dest[] = a_src[]
    return a_dest
end
function _permutedims(a::AbstractArray{<:Any, N}, perm::Tuple{Vararg{Int, N}}) where {N}
    return permutedims(a, perm)
end
function _permutedims(a::AbstractArray{<:Any, 0}, perm::Tuple{})
    return copy(a)
end
