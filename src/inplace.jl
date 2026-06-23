"""
    zero!(a::AbstractArray) -> a

In-place `zero`: set every entry of `a` to zero.
"""
zero!(a::AbstractArray) = (fill!(a, zero(eltype(a))); a)

"""
    scale!(a::AbstractArray, β::Number) -> a

In-place scaling: multiply every entry of `a` by `β`.
"""
scale!(a::AbstractArray, β::Number) = (a .*= β; a)
