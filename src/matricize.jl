using LinearAlgebra: Diagonal

using BlockArrays: AbstractBlockedUnitRange, blockedrange

using TensorProducts: ⊗
using .BaseExtensions: _permutedims, _permutedims!

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

struct ReshapeFusion <: FusionStyle end

FusionStyle(a::AbstractArray) = FusionStyle(a, axes(a))
function FusionStyle(a::AbstractArray, t::Tuple{Vararg{AbstractUnitRange}})
  return FusionStyle(a, combine_fusion_styles(FusionStyle.(t)...))
end

# Defaults to ReshapeFusion, a simple reshape
FusionStyle(::AbstractUnitRange) = ReshapeFusion()
FusionStyle(::AbstractArray, ::ReshapeFusion) = ReshapeFusion()

combine_fusion_styles() = ReshapeFusion()
combine_fusion_styles(::Style, ::Style) where {Style<:FusionStyle} = Style()
combine_fusion_styles(::FusionStyle, ::FusionStyle) = ReshapeFusion()
combine_fusion_styles(styles::FusionStyle...) = foldl(combine_fusion_styles, styles)

# =======================================  misc  ========================================
trivial_axis(::Tuple{}) = Base.OneTo(1)
trivial_axis(::Tuple{Vararg{AbstractUnitRange}}) = Base.OneTo(1)
trivial_axis(::Tuple{Vararg{AbstractBlockedUnitRange}}) = blockedrange([1])

function fuseaxes(
  axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockPermutation
)
  axesblocks = blocks(axes[blockedperm])
  return map(block -> isempty(block) ? trivial_axis(axes) : ⊗(block...), axesblocks)
end

# TODO remove _permutedims once support for Julia 1.10 is dropped
# define permutedims with a BlockedPermuation. Default is to flatten it.
function blockpermutedims(a::AbstractArray, biperm::AbstractBlockPermutation)
  return _permutedims(a, Tuple(biperm))
end

function blockpermutedims!(
  a::AbstractArray, b::AbstractArray, biperm::AbstractBlockPermutation
)
  return _permutedims!(a, b, Tuple(biperm))
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
  a_perm = blockpermutedims(a, biperm)
  return matricize(style, a_perm, trivialperm(biperm))
end

function matricize(
  style::FusionStyle, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  return throw(MethodError(matricize, Tuple{typeof(style),typeof(a),typeof(biperm)}))
end

# default is reshape
function matricize(::ReshapeFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2})
  new_axes = fuseaxes(axes(a), biperm)
  return reshape(a, new_axes...)
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
  return blockpermutedims(a_perm, invperm(biperm))
end

function unmatricize(::ReshapeFusion, m::AbstractMatrix, axes::AbstractUnitRange...)
  return reshape(m, axes...)
end

function unmatricize(
  ::ReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return reshape(m, Tuple(blocked_axes)...)
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
  return blockpermutedims!(a, a_perm, invperm(biperm))
end
