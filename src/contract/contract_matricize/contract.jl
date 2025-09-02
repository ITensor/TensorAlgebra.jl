using LinearAlgebra: mul!

function contract!(
  ::Matricize,
  a_dest::AbstractArray,
  biperm_a12_to_dest::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number,
  β::Number,
)
  biperm_dest_to_a12 = biperm(invperm(biperm_a12_to_dest), length_codomain(biperm1))

  check_input(contract, a_dest, biperm_dest_to_a12, a1, biperm1, a2, biperm2)
  a1_mat = matricize(a1, biperm1)
  a2_mat = matricize(a2, biperm2)
  a_dest_mat = a1_mat * a2_mat
  unmatricize_add!(a_dest, a_dest_mat, biperm_dest_to_a12, α, β)
  return a_dest
end
