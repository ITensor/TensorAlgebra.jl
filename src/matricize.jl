using TensorProducts: ⊗

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

struct ReshapeFusion <: FusionStyle end

FusionStyle(a::AbstractArray) = FusionStyle(a, axes(a))
function FusionStyle(a::AbstractArray, t::Tuple{Vararg{AbstractUnitRange}})
  return FusionStyle(a, combine_fusion_styles(FusionStyle.(t)...))
end

# Defaults to ReshapeFusion, a simple reshape
FusionStyle(::AbstractArray{<:Any,0}) = ReshapeFusion()   # TBD better solution?
FusionStyle(::AbstractUnitRange) = ReshapeFusion()
FusionStyle(::AbstractArray, ::ReshapeFusion) = ReshapeFusion()

combine_fusion_styles(::Style, ::Style) where {Style<:FusionStyle} = Style()
combine_fusion_styles(::FusionStyle, ::FusionStyle) = ReshapeFusion()
combine_fusion_styles(styles::FusionStyle...) = foldl(combine_fusion_styles, styles)

# =======================================  misc  ========================================
function fuseaxes(
  axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockPermutation
)
  axesblocks = blockpermute(axes, blockedperm)
  return map(block -> ⊗(block...), axesblocks)
end

Base.permutedims(a::AbstractArray, bp::AbstractBlockPermutation) = permutedims(a, Tuple(bp))
Base.permutedims(a::StridedArray, bp::AbstractBlockPermutation) = permutedims(a, Tuple(bp))

function Base.permutedims!(a::AbstractArray, b::AbstractArray, bp::AbstractBlockPermutation)
  return permutedims!(a, b, Tuple(bp))
end
function Base.permutedims!(
  a::Array{T,N}, b::StridedArray{T,N}, bp::AbstractBlockPermutation
) where {T,N}
  return permutedims!(a, b, Tuple(bp))
end

# =====================================  matricize  ========================================
# TBD settle copy/not copy convention
# matrix factorizations assume copy
# maybe: copy=false kwarg

# default is reshape
function matricize(
  ::ReshapeFusion,
  a::AbstractArray,
  row_axis::AbstractUnitRange,
  col_axis::AbstractUnitRange,
)
  return reshape(a, row_axis, col_axis)
end

function matricize(::ReshapeFusion, a::AbstractArray, bp::AbstractBlockPermutation{2})
  axes_fused = fuseaxes(axes(a), bp)
  return matricize(ReshapeFusion(), a, axes_fused...)
end

function matricize(a::AbstractArray, tp::BlockedTrivialPermutation{2})
  return matricize(FusionStyle(a), a, tp)
end

function matricize(a::AbstractArray, bp::AbstractBlockPermutation{2})
  a_perm = permutedims(a, bp)  # includes copy
  return matricize(a_perm, trivialperm(bp))
end

function matricize(a::AbstractArray, bt::AbstractBlockTuple{2})
  return matricize(a, blockedperm(bt))
end

function matricize(::AbstractArray, ::AbstractBlockTuple)
  throw(ArgumentError("Invalid axis permutation"))
end

function matricize(a::AbstractArray, permblocks...)
  return matricize(a, blockedpermvcat(permblocks...; length=Val(ndims(a))))
end

# ====================================  unmatricize  =======================================
function unmatricize(::ReshapeFusion, m::AbstractMatrix, axes::AbstractUnitRange...)
  return reshape(m, Base.to_shape.(axes)...)
end

function unmatricize(
  ::ReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return unmatricize(ReshapeFusion(), m, blocked_axes...)
end

function unmatricize(
  m::AbstractMatrix, blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}}
)
  return unmatricize(FusionStyle(m), m, blocked_axes)
end

function unmatricize(
  m::AbstractMatrix,
  codomain_axes::Tuple{Vararg{AbstractUnitRange}},
  domain_axes::Tuple{Vararg{AbstractUnitRange}},
)
  blocked_axes = tuplemortar((codomain_axes, domain_axes))
  return unmatricize(m, blocked_axes)
end

function unmatricize(
  m::AbstractMatrix, axes::Tuple{Vararg{AbstractUnitRange}}, bp::AbstractBlockPermutation{2}
)
  blocked_axes = tuplemortar(blockpermute(axes, bp))
  a_perm = unmatricize(m, blocked_axes)
  return permutedims(a_perm, invperm(bp))
end

function unmatricize!(a::AbstractArray, m::AbstractMatrix, bp::AbstractBlockPermutation{2})
  blocked_axes = tuplemortar(blockpermute(axes(a), bp))
  a_perm = unmatricize(m, blocked_axes)
  return permutedims!(a, a_perm, invperm(bp))
end
