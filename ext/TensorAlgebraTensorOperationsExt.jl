module TensorAlgebraTensorOperationsExt

using TensorAlgebra: TensorAlgebra, BlockedPermutation
using TupleTools
using TensorOperations
using TensorOperations: AbstractBackend as TOAlgorithm

TensorAlgebra.Algorithm(backend::TOAlgorithm) = backend

trivtuple(n) = ntuple(identity, n)

function _index2tuple(p::BlockedPermutation{2})
  N₁, N₂ = blocklengths(p)
  return (
    TupleTools.getindices(Tuple(p), trivtuple(N₁)),
    TupleTools.getindices(Tuple(p), N₁ .+ trivtuple(N₂)),
  )
end

# not in-place
# ------------
function TensorAlgebra.contract(
  backend::TOAlgorithm,
  pAB::BlockedPermutation,
  A::AbstractArray,
  pA::BlockedPermutation,
  B::AbstractArray,
  pB::BlockedPermutation,
  α::Number,
)
  pA′ = _index2tuple(pA)
  pB′ = _index2tuple(pB)
  pAB′ = _index2tuple(pAB)
  return tensorcontract(A, pA′, false, B, pB′, false, pAB′, α, backend)
end

function TensorAlgebra.contract(
  backend::TOAlgorithm,
  labelsC,
  A::AbstractArray,
  labelsA,
  B::AbstractArray,
  labelsB,
  α::Number,
)
  return tensorcontract(labelsC, A, labelsA, B, labelsB, α; backend)
end

# in-place
# --------
function TensorAlgebra.contract!(
  backend::TOAlgorithm,
  C::AbstractArray,
  pAB::BlockedPermutation,
  A::AbstractArray,
  pA::BlockedPermutation,
  B::AbstractArray,
  pB::BlockedPermutation,
  α::Number,
  β::Number,
)
  pA′ = _index2tuple(pA)
  pB′ = _index2tuple(pB)
  pAB′ = _index2tuple(pAB)
  return tensorcontract!(C, A, pA′, false, B, pB′, false, pAB′, α, β, backend)
end

function TensorAlgebra.contract!(
  backend::TOAlgorithm,
  C::AbstractArray,
  labelsC,
  A::AbstractArray,
  labelsA,
  B::AbstractArray,
  labelsB,
  α::Number,
  β::Number,
)
  pA, pB, pAB = TensorOperations.contract_indices(labelsA, labelsB, labelsC)
  return TensorOperations.tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend)
end

end
