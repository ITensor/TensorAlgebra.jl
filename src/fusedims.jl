using TensorProducts: ⊗
using .BaseExtensions: _permutedims, _permutedims!

abstract type FusionStyle end

struct ReshapeFusion <: FusionStyle end
struct BlockReshapeFusion <: FusionStyle end
struct SectorFusion <: FusionStyle end

# Defaults to a simple reshape
combine_fusion_styles(style1::Style, style2::Style) where {Style<:FusionStyle} = Style()
combine_fusion_styles(style1::FusionStyle, style2::FusionStyle) = ReshapeFusion()
combine_fusion_styles(styles::FusionStyle...) = foldl(combine_fusion_styles, styles)
FusionStyle(axis::AbstractUnitRange) = ReshapeFusion()
FusionStyle(::Tuple{}) = ReshapeFusion()
function FusionStyle(axes::Tuple{Vararg{AbstractUnitRange}})
  return combine_fusion_styles(FusionStyle.(axes)...)
end
FusionStyle(a::AbstractArray) = FusionStyle(axes(a))

# Overload this version for most arrays
function fusedims(::ReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...)
  return reshape(a, axes)
end

# Overload this version for most arrays
function fusedims(a::AbstractArray, ax::AbstractUnitRange, axes::AbstractUnitRange...)
  return fusedims(FusionStyle(a), a, ax, axes...)
end

function fusedims(
  a::AbstractArray,
  axb::Tuple{Vararg{AbstractUnitRange}},
  axesblocks::Tuple{Vararg{AbstractUnitRange}}...,
)
  return fusedims(a, flatten_tuples((axb, axesblocks...))...)
end

function fuseaxes(
  axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockPermutation
)
  axesblocks = blockpermute(axes, blockedperm)
  return map(block -> ⊗(block...), axesblocks)
end

function fuseaxes(a::AbstractArray, blockedperm::AbstractBlockPermutation)
  return fuseaxes(axes(a), blockedperm)
end

# Fuse adjacent dimensions
function fusedims(a::AbstractArray, blockedperm::BlockedTrivialPermutation)
  axes_fused = fuseaxes(a, blockedperm)
  return fusedims(a, axes_fused)
end

# deal with zero-dim case
fusedims(a::AbstractArray{<:Any,0}, t::Tuple{}...) = reshape(a, ntuple(_ -> 1, length(t)))

function fusedims(a::AbstractArray, bt::AbstractBlockTuple)
  # TBD define permutedims(::AbstractArray, ::AbstractBlockPermutation)
  # TBD remove call to BlockedTrivialPermutation?
  a_perm = _permutedims(a, Tuple(bt))
  return fusedims(a_perm, trivialperm(bt))
end

#  fusedims(ones((2,2,2,2)), (3, 1, 2), (4,))
#  fusedims(ones((2,2,2,2)), (3, 1, 2), 4)
function fusedims(a::AbstractArray, permblocks...)
  return fusedims(a, blockedpermvcat(permblocks...; length=Val(ndims(a))))
end
