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
  _mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  splitdims!(a_dest, a_dest_mat, biperm_dest)
  return a_dest
end

# Matrix multiplication.
function _mul!(
  a_dest::AbstractMatrix, a1::AbstractMatrix, a2::AbstractMatrix, α::Number, β::Number
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

# Inner product.
function _mul!(
  a_dest::AbstractArray{<:Any,0},
  a1::AbstractVector,
  a2::AbstractVector,
  α::Number,
  β::Number,
)
  a_dest[] = transpose(a1) * a2 * α + a_dest[] * β
  return a_dest
end

# Vec-mat.
function _mul!(
  a_dest::AbstractVector, a1::AbstractVector, a2::AbstractMatrix, α::Number, β::Number
)
  mul!(transpose(a_dest), transpose(a1), a2, α, β)
  return a_dest
end

# Mat-vec.
function _mul!(
  a_dest::AbstractVector, a1::AbstractMatrix, a2::AbstractVector, α::Number, β::Number
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

# Outer product.
function _mul!(
  a_dest::AbstractMatrix, a1::AbstractVector, a2::AbstractVector, α::Number, β::Number
)
  mul!(a_dest, a1, transpose(a2), α, β)
  return a_dest
end

# Array-scalar contraction.
function _mul!(
  a_dest::AbstractVector,
  a1::AbstractVector,
  a2::AbstractArray{<:Any,0},
  α::Number,
  β::Number,
)
  α′ = a2[] * α
  a_dest .= a1 .* α′ .+ a_dest .* β
  return a_dest
end

# Scalar-array contraction.
function _mul!(
  a_dest::AbstractVector,
  a1::AbstractArray{<:Any,0},
  a2::AbstractVector,
  α::Number,
  β::Number,
)
  # Preserve the ordering in case of non-commutative algebra.
  a_dest .= a1[] .* a2 .* α .+ a_dest .* β
  return a_dest
end

# Scalar-scalar contraction.
function _mul!(
  a_dest::AbstractArray{<:Any,0},
  a1::AbstractArray{<:Any,0},
  a2::AbstractArray{<:Any,0},
  α::Number,
  β::Number,
)
  # Preserve the ordering in case of non-commutative algebra.
  a_dest[] = a1[] * a2[] * α + a_dest[] * β
  return a_dest
end
