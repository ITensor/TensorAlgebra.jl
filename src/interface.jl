# TensorAlgebra's generic operations describe their operands in the `AbstractArray` vocabulary of
# `ndims` and `axes`. These are TensorAlgebra-owned functions, distinct from `Base.ndims`/
# `Base.axes`, that forward to Base by default. A backend for a non-`AbstractArray` tensor type
# (such as a `TensorMap`, whose "axes" are its index spaces) overloads these instead of committing
# type piracy on the `Base` functions.
function ndims end
ndims(a) = Base.ndims(a)

function axes end
axes(a) = Base.axes(a)
axes(a, i::Int) = Base.axes(a, i)
