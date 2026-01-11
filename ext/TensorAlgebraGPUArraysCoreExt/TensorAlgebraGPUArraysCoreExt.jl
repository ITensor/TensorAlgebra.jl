module TensorAlgebraGPUArraysCoreExt

using TensorAlgebra: TensorAlgebra, permutedimsadd!_view
using GPUArraysCore: AbstractGPUArray

# Overload to avoid converting to StridedView, which doesn't support GPU arrays.
function TensorAlgebra.permutedimsadd!(
        dest::AbstractGPUArray, src::AbstractGPUArray, perm, α::Number, β::Number
    )
    permutedimsadd!_view(dest, src, perm, α, β)
    return dest
end

end
