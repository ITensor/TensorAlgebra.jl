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
  axes_codomain, axes_contracted = blocks(axes(a1)[biperm1])
  axes_contracted2, axes_domain = blocks(axes(a2)[biperm2])
  @assert axes_contracted == axes_contracted2
  return genperm((axes_codomain..., axes_domain...), invperm(Tuple(biperm_dest)))
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
