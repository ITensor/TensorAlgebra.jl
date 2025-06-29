using Base.PermutedDimsArrays: genperm

function check_input(::typeof(contract), a1, labels1, a2, labels2)
  ndims(a1) == length(labels1) ||
    throw(ArgumentError("Invalid permutation for left tensor"))
  return ndims(a2) == length(labels2) ||
         throw(ArgumentError("Invalid permutation for right tensor"))
end

function check_input(::typeof(contract), a_dest, labels_dest, a1, labels1, a2, labels2)
  ndims(a_dest) == length(labels_dest) ||
    throw(ArgumentError("Invalid permutation for destination tensor"))
  return check_input(contract, a1, labels1, a2, labels2)
end

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
  check_input(contract, a1, biperm1, a2, biperm2)
  blocklengths(biperm_dest) == (length(biperm1[Block(1)]), length(biperm2[Block(2)])) ||
    throw(ArgumentError("Invalid permutation for destination tensor"))
  axes_dest = output_axes(contract, biperm_dest, a1, biperm1, a2, biperm2, α)
  return similar(a1, promote_type(eltype(a1), eltype(a2), typeof(α)), axes_dest)
end
