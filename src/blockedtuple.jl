# This file defines BlockedTuple, a Tuple of heterogeneous Tuple with a BlockArrays.jl
# like interface

using BlockArrays: Block, BlockArrays, BlockIndexRange, BlockRange, blockedrange

#
# ==================================  AbstractBlockTuple  ==================================
#
abstract type AbstractBlockTuple end

# Base interface
Base.axes(bt::AbstractBlockTuple) = (blockedrange([blocklengths(bt)...]),)

Base.copy(bt::AbstractBlockTuple) = copy.(bt)

Base.deepcopy(bt::AbstractBlockTuple) = deepcopy.(bt)

Base.firstindex(::AbstractBlockTuple) = 1

Base.getindex(bt::AbstractBlockTuple, i::Integer) = Tuple(bt)[i]
Base.getindex(bt::AbstractBlockTuple, r::AbstractUnitRange) = Tuple(bt)[r]
Base.getindex(bt::AbstractBlockTuple, b::Block{1}) = blocks(bt)[Int(b)]
Base.getindex(bt::AbstractBlockTuple, br::BlockRange{1}) = blocks(bt)[Int.(br)]
function Base.getindex(bt::AbstractBlockTuple, bi::BlockIndexRange{1})
  return bt[Block(bi)][only(bi.indices)]
end

Base.iterate(bt::AbstractBlockTuple) = iterate(Tuple(bt))
Base.iterate(bt::AbstractBlockTuple, i::Int) = iterate(Tuple(bt), i)

Base.length(bt::AbstractBlockTuple) = length(Tuple(bt))

Base.lastindex(bt::AbstractBlockTuple) = length(bt)

# Broadcast interface
Base.broadcastable(bt::AbstractBlockTuple) = bt
struct BlockedTupleBroadcastStyle{BlockLengths} <: Broadcast.BroadcastStyle end
function Base.BroadcastStyle(type::Type{<:AbstractBlockTuple})
  return BlockedTupleBroadcastStyle{blocklengths(type)}()
end

# BroadcastStyle is not called for two identical styles
function Base.BroadcastStyle(::BlockedTupleBroadcastStyle, ::BlockedTupleBroadcastStyle)
  throw(DimensionMismatch("Incompatible blocks"))
end
function Base.copy(
  bc::Broadcast.Broadcasted{BlockedTupleBroadcastStyle{BlockLengths}}
) where {BlockLengths}
  return BlockedTuple{BlockLengths}(bc.f.((Tuple.(bc.args))...))
end

# BlockArrays interface
function BlockArrays.blockfirsts(bt::AbstractBlockTuple)
  return (0, cumsum(Base.front(blocklengths(bt)))...) .+ 1
end

function BlockArrays.blocklasts(bt::AbstractBlockTuple)
  return cumsum(blocklengths(bt)[begin:end])
end

BlockArrays.blocklength(bt::AbstractBlockTuple) = length(blocklengths(bt))

BlockArrays.blocklengths(bt::AbstractBlockTuple) = blocklengths(typeof(bt))

function BlockArrays.blocks(bt::AbstractBlockTuple)
  bf = blockfirsts(bt)
  bl = blocklasts(bt)
  return ntuple(i -> Tuple(bt)[bf[i]:bl[i]], blocklength(bt))
end

#
# =====================================  BlockedTuple  =====================================
#
struct BlockedTuple{BlockLengths,Flat} <: AbstractBlockTuple
  flat::Flat

  function BlockedTuple{BlockLengths}(flat::Tuple) where {BlockLengths}
    length(flat) != sum(BlockLengths) && throw(DimensionMismatch("Invalid total length"))
    return new{BlockLengths,typeof(flat)}(flat)
  end
end

# TensorAlgebra Interface
BlockedTuple(tt::Vararg{Tuple}) = BlockedTuple{length.(tt)}(flatten_tuples(tt))
BlockedTuple(bt::AbstractBlockTuple) = BlockedTuple{blocklengths(bt)}(Tuple(bt))

# Base interface
Base.Tuple(bt::BlockedTuple) = bt.flat

Base.map(f, bt::BlockedTuple) = BlockedTuple{blocklengths(bt)}(map(f, Tuple(bt)))

# BlockArrays interface
function BlockArrays.blocklengths(::Type{<:BlockedTuple{BlockLengths}}) where {BlockLengths}
  return BlockLengths
end
