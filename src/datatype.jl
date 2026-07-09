"""
    data(a)

The underlying storage of `a`: the value reached by following `parent` to its fixed point
(an object that is its own `parent`). A wrapper returns the storage it ultimately wraps, and
a plain array returns itself.

# Examples

```jldoctest
julia> using TensorAlgebra: data

julia> a = [1.0 2.0; 3.0 4.0];

julia> data(transpose(a)) === a
true

julia> data(a) === a
true
```
"""
function data(a)
    parent_a = parent(a)
    parent_a === a && return a
    return data(parent_a)
end

"""
    datatype(a) -> Type

The type of the underlying storage of `a`, i.e. `typeof(data(a))`, in contrast to
`scalartype`/`eltype`, which give its element type alone.

# Examples

```jldoctest
julia> using TensorAlgebra: datatype

julia> datatype([1.0 2.0; 3.0 4.0])
Matrix{Float64} (alias for Array{Float64, 2})

julia> datatype(transpose([1.0 2.0; 3.0 4.0]))
Matrix{Float64} (alias for Array{Float64, 2})
```
"""
datatype(a) = typeof(data(a))
