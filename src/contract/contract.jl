# TODO: Add `contract!!` definitions as pass-throughs to `contract!`.

abstract type Algorithm end

Algorithm(alg::Algorithm) = alg

struct Matricize <: Algorithm end

default_contract_alg() = Matricize()

# Required interface if not using
# matricized contraction.
function contract!(
  alg::Algorithm,
  a_dest::AbstractArray,
  biperm_dest::AbstractBlockPermutation,
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation,
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation,
  α::Number,
  β::Number,
)
  return error("Not implemented")
end

function contract(
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number=one(Bool);
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), a1, labels1, a2, labels2, α; kwargs...)
end

function contract(
  alg::Algorithm,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number=one(Bool);
  kwargs...,
)
  labels_dest = output_labels(contract, alg, a1, labels1, a2, labels2, α; kwargs...)
  return contract(alg, labels_dest, a1, labels1, a2, labels2, α; kwargs...), labels_dest
end

function contract(
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number=one(Bool);
  alg=default_contract_alg(),
  kwargs...,
)
  return contract(Algorithm(alg), labels_dest, a1, labels1, a2, labels2, α; kwargs...)
end

function contract!(
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number=one(Bool),
  β::Number=zero(Bool);
  alg=default_contract_alg(),
  kwargs...,
)
  contract!(Algorithm(alg), a_dest, labels_dest, a1, labels1, a2, labels2, α, β; kwargs...)
  return a_dest
end

function contract(
  alg::Algorithm,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number=one(Bool);
  kwargs...,
)
  biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
  return contract(alg, biperm_dest, a1, biperm1, a2, biperm2, α; kwargs...)
end

function contract!(
  alg::Algorithm,
  a_dest::AbstractArray,
  labels_dest,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α::Number,
  β::Number;
  kwargs...,
)
  biperm_dest, biperm1, biperm2 = blockedperms(contract, labels_dest, labels1, labels2)
  return contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β; kwargs...)
end

function contract(
  alg::Algorithm,
  biperm_dest::AbstractBlockPermutation,
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation,
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation,
  α::Number;
  kwargs...,
)
  a_dest = allocate_output(contract, biperm_dest, a1, biperm1, a2, biperm2, α)
  contract!(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, zero(Bool); kwargs...)
  return a_dest
end
