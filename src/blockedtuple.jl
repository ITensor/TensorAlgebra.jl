# This file defines an abstract type AbstractBlockTuple and a concrete type BlockedTuple.
# These types allow to store a Tuple of heterogeneous Tuples with a BlockArrays.jl like
# interface.

using BlockArrays: Block, BlockArrays, BlockIndexRange, BlockRange, blockedrange

using TypeParameterAccessors: unspecify_type_parameters

#
# ==================================  AbstractBlockTuple  ==================================
#
# AbstractBlockTuple imposes BlockLength as first type parameter for easy dispatch
# it makes no assumotion on storage type
abstract type AbstractBlockTuple{BlockLength} end

constructorof(type::Type{<:AbstractBlockTuple}) = unspecify_type_parameters(type)
widened_constructorof(type::Type{<:AbstractBlockTuple}) = constructorof(type)

# Like `BlockRange`.
function blockeachindex(bt::AbstractBlockTuple)
  return ntuple(i -> Block(i), blocklength(bt))
end

# Base interface
Base.axes(bt::AbstractBlockTuple) = (blockedrange([blocklengths(bt)...]),)

Base.deepcopy(bt::AbstractBlockTuple) = deepcopy.(bt)

Base.firstindex(::AbstractBlockTuple) = 1

Base.getindex(bt::AbstractBlockTuple, i::Integer) = Tuple(bt)[i]
Base.getindex(bt::AbstractBlockTuple, r::AbstractUnitRange) = Tuple(bt)[r]
Base.getindex(bt::AbstractBlockTuple, b::Block{1}) = blocks(bt)[Int(b)]
function Base.getindex(bt::AbstractBlockTuple, br::BlockRange{1})
  r = Int.(br)
  flat = Tuple(bt)[blockfirsts(bt)[first(r)]:blocklasts(bt)[last(r)]]
  return widened_constructorof(typeof(bt))(flat, blocklengths(bt)[r])
end
function Base.getindex(bt::AbstractBlockTuple, bi::BlockIndexRange{1})
  return bt[Block(bi)][only(bi.indices)]
end

Base.iterate(bt::AbstractBlockTuple) = iterate(Tuple(bt))
Base.iterate(bt::AbstractBlockTuple, i::Int) = iterate(Tuple(bt), i)

Base.lastindex(bt::AbstractBlockTuple) = length(bt)

Base.length(bt::AbstractBlockTuple) = sum(blocklengths(bt); init=0)

function Base.map(f, bt::AbstractBlockTuple)
  BL = blocklengths(bt)
  # use Val to preserve compile time knowledge of BL
  return widened_constructorof(typeof(bt))(map(f, Tuple(bt)), Val(BL))
end

# Broadcast interface
Base.broadcastable(bt::AbstractBlockTuple) = bt
struct AbstractBlockTupleBroadcastStyle{BlockLengths,BT} <: Broadcast.BroadcastStyle end
function Base.BroadcastStyle(T::Type{<:AbstractBlockTuple})
  return AbstractBlockTupleBroadcastStyle{blocklengths(T),unspecify_type_parameters(T)}()
end

# BroadcastStyle is not called for two identical styles
function Base.BroadcastStyle(
  ::AbstractBlockTupleBroadcastStyle, ::AbstractBlockTupleBroadcastStyle
)
  throw(DimensionMismatch("Incompatible blocks"))
end
function Base.copy(
  bc::Broadcast.Broadcasted{AbstractBlockTupleBroadcastStyle{BlockLengths,BT}}
) where {BlockLengths,BT}
  return widened_constructorof(BT)(bc.f.((Tuple.(bc.args))...), Val(BlockLengths))
end

Base.ndims(::Type{<:AbstractBlockTuple}) = 1  # needed in nested broadcast

# BlockArrays interface
BlockArrays.blockfirsts(::AbstractBlockTuple{0}) = ()
function BlockArrays.blockfirsts(bt::AbstractBlockTuple)
  return (0, cumsum(Base.front(blocklengths(bt)))...) .+ 1
end

function BlockArrays.blocklasts(bt::AbstractBlockTuple)
  return cumsum(blocklengths(bt))
end

BlockArrays.blocklength(::AbstractBlockTuple{BlockLength}) where {BlockLength} = BlockLength

BlockArrays.blocklengths(bt::AbstractBlockTuple) = blocklengths(typeof(bt))

function BlockArrays.blocks(bt::AbstractBlockTuple)
  bf = blockfirsts(bt)
  bl = blocklasts(bt)
  return ntuple(i -> Tuple(bt)[bf[i]:bl[i]], blocklength(bt))
end

#    length(BlockLengths) != BlockLength && throw(DimensionMismatch("Invalid blocklength"))

# =====================================  BlockedTuple  =====================================
#
struct BlockedTuple{BlockLength,BlockLengths,Flat} <: AbstractBlockTuple{BlockLength}
  flat::Flat

  function BlockedTuple{BlockLength,BlockLengths}(
    flat::Tuple
  ) where {BlockLength,BlockLengths}
    length(BlockLengths) != BlockLength && throw(DimensionMismatch("Invalid blocklength"))
    length(flat) != sum(BlockLengths; init=0) &&
      throw(DimensionMismatch("Invalid total length"))
    any(BlockLengths .< 0) && throw(DimensionMismatch("Invalid block length"))
    return new{BlockLength,BlockLengths,typeof(flat)}(flat)
  end
end

# TensorAlgebra Interface
function tuplemortar(tt::Tuple{Vararg{Tuple}})
  return BlockedTuple{length(tt),length.(tt)}(flatten_tuples(tt))
end
function BlockedTuple(flat::Tuple, BlockLengths::Tuple{Vararg{Int}})
  return BlockedTuple{length(BlockLengths),BlockLengths}(flat)
end
function BlockedTuple(flat::Tuple, ::Val{BlockLengths}) where {BlockLengths}
  # use Val to preserve compile time knowledge of BL
  return BlockedTuple{length(BlockLengths),BlockLengths}(flat)
end
function BlockedTuple(bt::AbstractBlockTuple)
  bl = blocklengths(bt)
  return BlockedTuple{length(bl),bl}(Tuple(bt))
end

# Base interface
Base.Tuple(bt::BlockedTuple) = bt.flat

# BlockArrays interface
function BlockArrays.blocklengths(
  ::Type{<:BlockedTuple{<:Any,BlockLengths}}
) where {BlockLengths}
  return BlockLengths
end
