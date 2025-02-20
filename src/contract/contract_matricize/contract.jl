using LinearAlgebra: mul!

function contract!(
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
  a_dest_mat = fusedims(a_dest, biperm_dest)
  a1_mat = fusedims(a1, biperm1)
  a2_mat = fusedims(a2, biperm2)
  @assert ndims(a1_mat) == 2
  @assert ndims(a2_mat) == 2
  mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  splitdims!(a_dest, a_dest_mat, biperm_dest)
  return a_dest
end
