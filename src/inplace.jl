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

"""
    twist!(a::AbstractArray, dims) -> a

In-place twist of `a` over the axes in `dims`, which scales `a` by the product of the sector
twists of those axes (`-1` for odd-parity fermionic charges, `+1` otherwise). For an array
without sector data, such as a plain `Array`, this is a no-op.
"""
twist!(a::AbstractArray, dims) = a
