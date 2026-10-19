module TensorAlgebraGPUArraysCoreExt

using GPUArraysCore: @allowscalar, AnyGPUArray
using TensorAlgebra: TensorAlgebra

# `a[]` on a GPU array is a scalar index, which GPU array packages disallow by default.
TensorAlgebra.scalar(a::AnyGPUArray{<:Any, 0}) = @allowscalar a[]

end
