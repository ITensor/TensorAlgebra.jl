module TensorAlgebraBlockSparseArraysGradedUnitRangesExt

using BlockArrays: Block, blocksize
using BlockSparseArrays: BlockSparseMatrix, @view!
using GradedUnitRanges: AbstractGradedUnitRange, dual, tensor_product
using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra, random_unitary!

function TensorAlgebra.:⊗(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  return tensor_product(a1, a2)
end

function TensorAlgebra.square_zero_map(
  elt::Type, ax::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}
)
  return BlockSparseArray{elt}(undef, (dual.(ax)..., ax...))
end

function TensorAlgebra.random_unitary!(
  rng::AbstractRNG,
  a::BlockSparseMatrix{
    <:Any,<:Any,<:Any,<:Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}
  },
)
  # TODO: Define and use `blockdiagindices`
  # or `blockdiaglength`.
  for i in 1:blocksize(a, 1)
    random_unitary!(rng, @view!(a[Block(i, i)]))
  end
  return a
end

end
