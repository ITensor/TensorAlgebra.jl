using LinearAlgebra: mul!

function contractadd!(
  ::Matricize,
  a_dest::AbstractArray,
  biperm_dest::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  check_input(contract, a_dest, biperm_dest, a1, biperm1, a2, biperm2)
  a_dest_mat = matricize(a_dest, biperm_dest)
  a1_mat = matricize(a1, biperm1)
  a2_mat = matricize(a2, biperm2)
  mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  unmatricize!(a_dest, a_dest_mat, biperm_dest)
  return a_dest
end
