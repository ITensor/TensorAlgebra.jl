using LinearAlgebra: Diagonal

using BlockArrays: AbstractBlockedUnitRange, blockedrange

using TensorProducts: ⊗
using .BaseExtensions: _permutedims, _permutedims!

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

struct ReshapeFusion <: FusionStyle end

FusionStyle(x) = FusionStyle(typeof(x))
FusionStyle(T::Type) = throw(MethodError(FusionStyle, (T,)))

# Defaults to ReshapeFusion, a simple reshape
FusionStyle(::Type{<:AbstractArray}) = ReshapeFusion()

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
function permuteblockeddims(a::AbstractArray, biperm::AbstractBlockPermutation)
  return _permutedims(a, Tuple(biperm))
end

function permuteblockeddims!(
  a::AbstractArray, b::AbstractArray, biperm::AbstractBlockPermutation
)
  return _permutedims!(a, b, Tuple(biperm))
end

# =====================================  matricize  ========================================
# TBD settle copy/not copy convention
# matrix factorizations assume copy
# maybe: copy=false kwarg

function matricize(a::AbstractArray, biperm_dest::AbstractBlockPermutation{2})
  ndims(a) == length(biperm_dest) || throw(ArgumentError("Invalid bipermutation"))
  return matricize(FusionStyle(a), a, biperm_dest)
end

function matricize(
  style::FusionStyle, a::AbstractArray, biperm_dest::AbstractBlockPermutation{2}
)
  a_perm = permuteblockeddims(a, biperm_dest)
  return matricize(style, a_perm, trivialperm(biperm_dest))
end

function matricize(
  style::FusionStyle, a::AbstractArray, biperm_dest::BlockedTrivialPermutation{2}
)
  return throw(MethodError(matricize, Tuple{typeof(style),typeof(a),typeof(biperm_dest)}))
end

# default is reshape
function matricize(
  ::ReshapeFusion, a::AbstractArray, biperm_dest::BlockedTrivialPermutation{2}
)
  new_axes = fuseaxes(axes(a), biperm_dest)
  return reshape(a, new_axes...)
end

function matricize(a::AbstractArray, permblock1::Tuple, permblock2::Tuple)
  return matricize(a, blockedpermvcat(permblock1, permblock2; length=Val(ndims(a))))
end

# ====================================  unmatricize  =======================================
function unmatricize(m::AbstractMatrix, axes_dest, invbiperm::AbstractBlockPermutation{2})
  length(axes_dest) == length(invbiperm) ||
    throw(ArgumentError("axes do not match permutation"))
  return unmatricize(FusionStyle(m), m, axes_dest, invbiperm)
end

function unmatricize(
  ::FusionStyle, m::AbstractMatrix, axes_dest, invbiperm::AbstractBlockPermutation{2}
)
  blocked_axes = axes_dest[invbiperm]
  a12 = unmatricize(m, blocked_axes)
  biperm_dest = biperm(invperm(invbiperm), length_codomain(axes_dest))

  return permuteblockeddims(a12, biperm_dest)
end

function unmatricize(
  ::ReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return reshape(m, Tuple(blocked_axes)...)
end

function unmatricize(m::AbstractMatrix, blocked_axes)
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

function unmatricize!(a_dest, m::AbstractMatrix, invbiperm::AbstractBlockPermutation{2})
  ndims(a_dest) == length(invbiperm) ||
    throw(ArgumentError("destination does not match permutation"))
  blocked_axes = axes(a_dest)[invbiperm]
  a_perm = unmatricize(m, blocked_axes)
  biperm_dest = biperm(invperm(invbiperm), length_codomain(axes(a_dest)))
  return permuteblockeddims!(a_dest, a_perm, biperm_dest)
end

function unmatricizeadd!(a_dest, a_dest_mat, invbiperm, α, β)
  a12 = unmatricize(a_dest_mat, axes(a_dest), invbiperm)
  a_dest .= α .* a12 .+ β .* a_dest
  return a_dest
end
