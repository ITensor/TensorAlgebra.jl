module TensorAlgebraTensorOperationsExt

using TensorAlgebra: TensorAlgebra, BlockedPermutation, Algorithm
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

_blockedpermutation(p::Index2Tuple) = TensorAlgebra.blockedpermvcat(p...)

# Using TensorAlgebra implementations as TensorOperations backends
# ----------------------------------------------------------------
function TensorOperations.tensorcontract!(
  C::AbstractArray,
  A::AbstractArray,
  pA::Index2Tuple,
  conjA::Bool,
  B::AbstractArray,
  pB::Index2Tuple,
  conjB::Bool,
  pAB::Index2Tuple,
  α::Number,
  β::Number,
  backend::Algorithm,
  allocator,
)
  bipermA = _blockedpermutation(pA)
  bipermB = _blockedpermutation(pB)
  bipermAB = _blockedpermutation(pAB)
  A′ = conjA ? conj(A) : A
  B′ = conjB ? conj(B) : B
  return TensorAlgebra.contract!(backend, C, bipermAB, A′, bipermA, B′, bipermB, α, β)
end

