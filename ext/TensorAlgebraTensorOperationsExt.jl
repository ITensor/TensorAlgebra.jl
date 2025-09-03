module TensorAlgebraTensorOperationsExt

using TensorAlgebra: TensorAlgebra, BlockedPermutation, Algorithm
using TupleTools
using TensorOperations
using TensorOperations: AbstractBackend, DefaultBackend

"""
    TensorOperationsAlgorithm(backend::AbstractBackend)

Wrapper type for making a TensorOperations backend work as a TensorAlgebra algorithm.
"""
struct TensorOperationsAlgorithm{B<:AbstractBackend} <: Algorithm
  backend::B
end

TensorAlgebra.Algorithm(backend::AbstractBackend) = TensorOperationsAlgorithm(backend)

trivtuple(n) = ntuple(identity, n)

function _index2tuple(p::BlockedPermutation{2})
  N₁, N₂ = blocklengths(p)
  return (
    TupleTools.getindices(Tuple(p), trivtuple(N₁)),
    TupleTools.getindices(Tuple(p), N₁ .+ trivtuple(N₂)),
  )
end

_blockedpermutation(p::Index2Tuple) = TensorAlgebra.blockedpermvcat(p...)

# Using TensorOperations backends as TensorAlgebra implementations
# ----------------------------------------------------------------

# not in-place
function TensorAlgebra.contract(
  algorithm::TensorOperationsAlgorithm,
  bipermAB::BlockedPermutation,
  A::AbstractArray,
  bipermA::BlockedPermutation,
  B::AbstractArray,
  bipermB::BlockedPermutation,
  α::Number,
)
  pA = _index2tuple(bipermA)
  pB = _index2tuple(bipermB)
  pAB = _index2tuple(bipermAB)

  return tensorcontract(A, pA, false, B, pB, false, pAB, α, algorithm.backend)
end

function TensorAlgebra.contract(
  algorithm::TensorOperationsAlgorithm,
  labelsC,
  A::AbstractArray,
  labelsA,
  B::AbstractArray,
  labelsB,
  α::Number,
)
  pA, pB, pAB = TensorOperations.contract_indices(labelsA, labelsB, labelsC)
  return tensorcontract(A, pA, false, B, pB, false, pAB, α, algorithm.backend)
end

# in-place
function TensorAlgebra.contract!(
  algorithm::TensorOperationsAlgorithm,
  C::AbstractArray,
  bipermAB::BlockedPermutation,
  A::AbstractArray,
  bipermA::BlockedPermutation,
  B::AbstractArray,
  bipermB::BlockedPermutation,
  α::Number,
  β::Number,
)
  pA = _index2tuple(bipermA)
  pB = _index2tuple(bipermB)
  pAB = _index2tuple(bipermAB)
  return tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, algorithm.backend)
end

function TensorAlgebra.contract!(
  algorithm::TensorOperationsAlgorithm,
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
  return TensorOperations.tensorcontract!(
    C, A, pA, false, B, pB, false, pAB, α, β, algorithm.backend
  )
end

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

# For now no trace/add is supported, so simply reselect default backend from TensorOperations
function TensorOperations.tensortrace!(
  C::AbstractArray,
  A::AbstractArray,
  p::Index2Tuple,
  q::Index2Tuple,
  conjA::Bool,
  α::Number,
  β::Number,
  ::Algorithm,
  allocator,
)
  return TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, DefaultBackend(), allocator)
end
function TensorOperations.tensoradd!(
  C::AbstractArray,
  A::AbstractArray,
  pA::Index2Tuple,
  conjA::Bool,
  α::Number,
  β::Number,
  ::Algorithm,
  allocator,
)
  return TensorOperations.tensoradd!(C, A, pA, conjA, α, β, DefaultBackend(), allocator)
end

end
