# Generic in-place array primitives that TensorAlgebra owns and downstream array types
# extend. `zero!` was previously provided by FunctionImplementations; `scale!` was a
# generic fallback that lived in GradedArrays. Both belong here, with TensorAlgebra as
# the home of the in-place tensor-algebra interface.

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
