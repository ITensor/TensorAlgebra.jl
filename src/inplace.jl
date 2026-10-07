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

In-place ribbon twist over the axes in `dims`. A plain array carries no sector data and has no
braiding, so this is a no-op; graded-array backends override it to scale `a` by the product of
the sector twists of those axes (`-1` for odd-parity fermionic charges, `+1` otherwise).
"""
twist!(a::AbstractArray, dims) = a
