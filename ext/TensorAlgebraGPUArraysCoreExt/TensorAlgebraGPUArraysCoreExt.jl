module TensorAlgebraGPUArraysCoreExt

import TensorAlgebra as TA
using GPUArraysCore: AbstractGPUArray

# Overload to avoid converting to StridedView, which doesn't support GPU arrays.
function TA.add!(dest::AbstractGPUArray, src::AbstractGPUArray, α::Number, β::Number)
    return TA._add!(dest, src, perm, α, β)
end

end
