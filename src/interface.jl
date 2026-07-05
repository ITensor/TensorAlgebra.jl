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

# Eager conjugation in the same vocabulary: conjugated elements on conjugated (dualized)
# axes, leg order unchanged. Backends whose native conjugation is `adjoint`-shaped (such as
# a `TensorMap`, where the adjoint also swaps codomain and domain) overload this to restore
# the original leg order. Named `conjugate` (the eager companion of the lazy `conjed`)
# rather than `conj`: `Base.conj` is passed around as an `op` value inside this package
# (`op === conj` guards, `::typeof(conj)` dispatch), so a module-level `conj` binding
# would silently change what those mean.
function conjugate end
conjugate(a) = Base.conj(a)

# The scalar held by a rank-0 tensor. The Base spelling is `a[]`, which a `TensorMap`
# with a nontrivial sector type does not support (TensorKit provides `scalar` instead).
function scalar end
scalar(a) = a[]

# The sum of the (dense) elements. A `TensorMap` is not iterable, so `Base.sum` does not
# apply to it directly.
function sum end
sum(a; kwargs...) = Base.sum(a; kwargs...)
