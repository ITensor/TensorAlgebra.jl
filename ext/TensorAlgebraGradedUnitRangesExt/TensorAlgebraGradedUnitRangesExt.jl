module TensorAlgebraGradedUnitRangesExt

using GradedUnitRanges: AbstractGradedUnitRange, dual, tensor_product
using GradedUnitRanges.BlockArrays: Block, blocklengths, blocksize
using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra, random_unitary

function TensorAlgebra.:⊗(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  return tensor_product(a1, a2)
end

function TensorAlgebra.dual(a::AbstractGradedUnitRange)
  return dual(a)
end

function TensorAlgebra.random_unitary(
  rng::AbstractRNG,
  elt::Type,
  ax::Tuple{AbstractGradedUnitRange},
)
  a = zeros(elt, dual.(ax)..., ax...)
  # TODO: Define `blockdiagindices`.
  for i in 1:minimum(blocksize(a))
    a[Block(i, i)] = random_unitary(rng, elt, Int(blocklengths(only(ax))[i]))
  end
  return a
end

end
