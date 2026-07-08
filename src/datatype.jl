"""
    datatype(a) -> Type
    datatype(::Type) -> Type

The underlying storage array type of `a`, capturing both the element type and the
container/device (e.g. `Vector{Float64}`, `CuArray{ComplexF32}`), as opposed to
`scalartype`/`eltype` which capture the element type alone.

The instance form is primary: array wrappers recurse through `parent` (the same
unwrapping Adapt.jl uses), so `Diagonal`, `SubArray`, `transpose`, `Adjoint`, and
similar wrappers resolve to their underlying storage with no bespoke method. A plain
array is its own storage (its `parent` is itself), which terminates the recursion. The
type form is the base case `datatype(::Type{T}) where {T<:AbstractArray} = T` and does
not unwrap, since a wrapper's backing is generally not recoverable from its type alone.

Backends whose backing is reached through an instance operation (e.g. an ITensor via
`unnamed`, a `TensorMap` via its fused data) add their own `datatype` overloads.

# Examples

```jldoctest
julia> using TensorAlgebra: datatype

julia> datatype(randn(2, 3))
Matrix{Float64} (alias for Array{Float64, 2})

julia> datatype(transpose(randn(2, 3)))
Matrix{Float64} (alias for Array{Float64, 2})
```
"""
function datatype end

datatype(type::Type{<:AbstractArray}) = type

function datatype(a::AbstractArray)
    parent_a = parent(a)
    parent_a === a && return typeof(a)
    return datatype(parent_a)
end
