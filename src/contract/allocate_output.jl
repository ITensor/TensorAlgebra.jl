using Base.PermutedDimsArrays: genperm

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function output_axes(
  ::typeof(contract),
  biperm_dest::AbstractBlockPermutation{2},
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number=one(Bool),
)
  axes_codomain, axes_contracted = blockpermute(axes(a1), biperm1)
  axes_contracted2, axes_domain = blockpermute(axes(a2), biperm2)
  @assert axes_contracted == axes_contracted2
  return genperm((axes_codomain..., axes_domain...), invperm(Tuple(biperm_dest)))
end

# Inner-product contraction.
# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{0},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{1},
  a2::AbstractArray,
  perm2::AbstractBlockPermutation{1},
  α::Number=one(Bool),
)
  axes_contracted = blockpermute(axes(a1), perm1)
  axes_contracted′ = blockpermute(axes(a2), perm2)
  @assert axes_contracted == axes_contracted′
  return ()
end

# Vec-mat.
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{1},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{1},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{2},
  α::Number=one(Bool),
)
  (axes_contracted,) = blockpermute(axes(a1), perm1)
  axes_contracted′, axes_dest = blockpermute(axes(a2), biperm2)
  @assert axes_contracted == axes_contracted′
  return genperm((axes_dest...,), invperm(Tuple(perm_dest)))
end

# Mat-vec.
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{1},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{2},
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation{1},
  α::Number=one(Bool),
)
  axes_dest, axes_contracted = blockpermute(axes(a1), perm1)
  (axes_contracted′,) = blockpermute(axes(a2), biperm2)
  @assert axes_contracted == axes_contracted′
  return genperm((axes_dest...,), invperm(Tuple(perm_dest)))
end

# Outer product.
function output_axes(
  ::typeof(contract),
  biperm_dest::AbstractBlockPermutation{2},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{1},
  a2::AbstractArray,
  perm2::AbstractBlockPermutation{1},
  α::Number=one(Bool),
)
  @assert istrivialperm(Tuple(perm1))
  @assert istrivialperm(Tuple(perm2))
  axes_dest = (axes(a1)..., axes(a2)...)
  return genperm(axes_dest, invperm(Tuple(biperm_dest)))
end

# Array-scalar contraction.
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{1},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{1},
  a2::AbstractArray,
  perm2::AbstractBlockPermutation{0},
  α::Number=one(Bool),
)
  @assert istrivialperm(Tuple(perm1))
  axes_dest = axes(a1)
  return genperm(axes_dest, invperm(Tuple(perm_dest)))
end

# Scalar-array contraction.
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{1},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{0},
  a2::AbstractArray,
  perm2::AbstractBlockPermutation{1},
  α::Number=one(Bool),
)
  @assert istrivialperm(Tuple(perm2))
  axes_dest = axes(a2)
  return genperm(axes_dest, invperm(Tuple(perm_dest)))
end

# Scalar-scalar contraction.
function output_axes(
  ::typeof(contract),
  perm_dest::AbstractBlockPermutation{0},
  a1::AbstractArray,
  perm1::AbstractBlockPermutation{0},
  a2::AbstractArray,
  perm2::AbstractBlockPermutation{0},
  α::Number=one(Bool),
)
  return ()
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function allocate_output(
  ::typeof(contract),
  biperm_dest::AbstractBlockPermutation,
  a1::AbstractArray,
  biperm1::AbstractBlockPermutation,
  a2::AbstractArray,
  biperm2::AbstractBlockPermutation,
  α::Number=one(Bool),
)
  axes_dest = output_axes(contract, biperm_dest, a1, biperm1, a2, biperm2, α)
  return similar(a1, promote_type(eltype(a1), eltype(a2), typeof(α)), axes_dest)
end
