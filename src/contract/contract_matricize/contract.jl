using LinearAlgebra: mul!

function contract!(
  ::Matricize,
  a_dest::AbstractArray,
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  biperm_out = blockedtrivialperm((length(biperm1[Block(1)]), length(biperm2[Block(2)])))
  check_input(contract, a_dest, biperm_out, a1, biperm1, a2, biperm2)
  a_dest_mat = matricize(a_dest, biperm_out)
  a1_mat = matricize(a1, biperm1)
  a2_mat = matricize(a2, biperm2)
  mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  unmatricize!(a_dest, a_dest_mat, biperm_out)
  return a_dest
end
