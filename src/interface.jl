# TensorAlgebra's generic operations describe their operands in the `AbstractArray` vocabulary of
# `ndims`, `axes`, and `size`. These are TensorAlgebra-owned functions, distinct from `Base.ndims`/
# `Base.axes`/`Base.size`, that forward to Base by default. A backend for a non-`AbstractArray`
# tensor type (such as a `TensorMap`, whose "axes" are its index spaces) overloads these instead of
# committing type piracy on the `Base` functions.
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

The single scalar held by a rank-0 (zero-dimensional) `a`, i.e. `a[]`.
"""
function scalar end
# The Base spelling is `a[]`, which a `TensorMap` with a nontrivial sector type does not
# support (TensorKit provides `scalar` instead).
scalar(a) = a[]

# The sum of the (dense) elements. A `TensorMap` is not iterable, so `Base.sum` does not
# apply to it directly.
function sum end
sum(a; kwargs...) = Base.sum(a; kwargs...)
