module TensorAlgebraGPUArraysCoreExt

import TensorAlgebra as TA
using GPUArraysCore: AnyGPUArray

TA.iscpu(::AnyGPUArray) = false

end
