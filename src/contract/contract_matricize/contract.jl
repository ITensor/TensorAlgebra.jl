using LinearAlgebra: mul!

function isinplace(::AbstractArray, biperm_out)
  return istrivialperm(Tuple(biperm_out))
end

function contract!(
  alg::Matricize,
  a_dest::AbstractArray,
  biperm_out::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  if isinplace(a_dest, biperm_out)
    return contract_inplace!(alg, a_dest, biperm_out, a1, biperm1, a2, biperm2, α, β)
  else
    return contract_outofplace!(alg, a_dest, biperm_out, a1, biperm1, a2, biperm2, α, β)
  end
end

function contract_inplace!(
  ::Matricize,
  a_dest::AbstractArray,
  biperm_out::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  biperm_dest = invbiperm(biperm_out, Val(first(blocklengths(biperm1))))
  check_input(contract, a_dest, biperm_dest, a1, biperm1, a2, biperm2)
  a_dest_mat = matricize(a_dest, biperm_dest)
  a1_mat = matricize(a1, biperm1)
  a2_mat = matricize(a2, biperm2)
  mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  unmatricize!(a_dest, a_dest_mat, biperm_dest)  # TODO remove: need no copy in matricize
  return a_dest
end

function contract_outofplace!(
  ::Matricize,
  a_dest::AbstractArray,
  biperm_out::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  biperm_dest = invbiperm(biperm_out, Val(first(blocklengths(biperm1))))
  check_input(contract, a_dest, biperm_dest, a1, biperm1, a2, biperm2)
  a1_mat = matricize(a1, biperm1)
  a2_mat = matricize(a2, biperm2)
  a_dest_mat = a1_mat * a2_mat
  unmatricize_add!(a_dest, a_dest_mat, biperm_dest, α, β)
  return a_dest
end
