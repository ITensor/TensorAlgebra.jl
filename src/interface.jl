# Owned by TensorAlgebra and forwarding to Base, so a non-`AbstractArray` tensor type (such as a
# `TensorMap`) can overload them without type piracy.
function ndims end
ndims(a) = Base.ndims(a)

function axes end
axes(a) = Base.axes(a)
axes(a, i::Int) = Base.axes(a, i)

function size end
size(a) = Base.size(a)
size(a, i::Int) = Base.size(a, i)

"""
    scalar(a)

The single scalar held by a zero-dimensional `a`, the equivalent of `a[]`.

# Examples

```jldoctest
julia> import TensorAlgebra

julia> TensorAlgebra.scalar(fill(3.0))
3.0
```
"""
function scalar end
# The Base spelling is `a[]`, which a `TensorMap` with a nontrivial sector type does not
# support (TensorKit provides `scalar` instead).
scalar(a) = a[]

# The sum of the (dense) elements. A `TensorMap` is not iterable, so `Base.sum` does not
# apply to it directly.
function sum end
sum(a; kwargs...) = Base.sum(a; kwargs...)
