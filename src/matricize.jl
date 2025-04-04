using LinearAlgebra: Diagonal

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
  axesblocks = blocks(axes[blockedperm])
  return map(block -> ⊗(block...), axesblocks)
end

# define permutedims with a BlockedPermuation. Default is to flatten it.
function Base.permutedims(a::AbstractArray, biperm::AbstractBlockPermutation)
  return permutedims(a, Tuple(biperm))
end

# solve ambiguities
function Base.permutedims(a::StridedArray, biperm::AbstractBlockPermutation)
  return permutedims(a, Tuple(biperm))
end
function Base.permutedims(a::Diagonal, biperm::AbstractBlockPermutation)
  return permutedims(a, Tuple(biperm))
end

function Base.permutedims!(
  a::AbstractArray, b::AbstractArray, biperm::AbstractBlockPermutation
)
  return permutedims!(a, b, Tuple(biperm))
end

# solve ambiguities
function Base.permutedims!(
  a::Array{T,N}, b::StridedArray{T,N}, biperm::AbstractBlockPermutation
) where {T,N}
  return permutedims!(a, b, Tuple(biperm))
end

# =====================================  matricize  ========================================
# TBD settle copy/not copy convention
# matrix factorizations assume copy
# maybe: copy=false kwarg

function matricize(a::AbstractArray, biperm::AbstractBlockPermutation{2})
  return matricize(FusionStyle(a), a, biperm)
end

function matricize(
  style::FusionStyle, a::AbstractArray, biperm::AbstractBlockPermutation{2}
)
  a_perm = permutedims(a, biperm)
  return matricize(style, a_perm, trivialperm(biperm))
end

function matricize(
  style::FusionStyle, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  return throw(MethodError(matricize, Tuple{typeof(style),typeof(a),typeof(biperm)}))
end

# default is reshape
function matricize(::ReshapeFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2})
  return reshape(a, fuseaxes(axes(a), biperm)...)
end

function matricize(a::AbstractArray, bt::AbstractBlockTuple{2})
  return matricize(a, blockedperm(bt))
end

function matricize(::AbstractArray, ::AbstractBlockTuple)
  throw(ArgumentError("Invalid axis permutation"))
end

function matricize(a::AbstractArray, permblock1::Tuple, permblock2::Tuple)
  return matricize(a, blockedpermvcat(permblock1, permblock2; length=Val(ndims(a))))
end

# ====================================  unmatricize  =======================================
function unmatricize(
  m::AbstractMatrix,
  axes::Tuple{Vararg{AbstractUnitRange}},
  biperm::AbstractBlockPermutation{2},
)
  return unmatricize(FusionStyle(m), m, axes, biperm)
end

function unmatricize(
  ::FusionStyle,
  m::AbstractMatrix,
  axes::Tuple{Vararg{AbstractUnitRange}},
  biperm::AbstractBlockPermutation{2},
)
  blocked_axes = axes[biperm]
  a_perm = unmatricize(m, blocked_axes)
  return permutedims(a_perm, invperm(biperm))
end

function unmatricize(::ReshapeFusion, m::AbstractMatrix, axes::AbstractUnitRange...)
  return reshape(m, Base.to_shape.(axes)...)
end

function unmatricize(
  ::ReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return reshape(m, Base.to_shape.(Tuple(blocked_axes))...)
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

function unmatricize!(
  a::AbstractArray, m::AbstractMatrix, biperm::AbstractBlockPermutation{2}
)
  blocked_axes = axes(a)[biperm]
  a_perm = unmatricize(m, blocked_axes)
  return permutedims!(a, a_perm, invperm(biperm))
end
