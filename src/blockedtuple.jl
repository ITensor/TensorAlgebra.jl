# This file defines BlockedTuple, a Tuple of heterogeneous Tuple with a BlockArrays.jl
# like interface

using BlockArrays: Block, BlockArrays, BlockIndexRange, BlockRange, blockedrange

using TypeParameterAccessors: unspecify_type_parameters

#
# ==================================  AbstractBlockTuple  ==================================
#
abstract type AbstractBlockTuple end

# Base interface
Base.axes(bt::AbstractBlockTuple) = (blockedrange([blocklengths(bt)...]),)

Base.deepcopy(bt::AbstractBlockTuple) = deepcopy.(bt)

Base.firstindex(::AbstractBlockTuple) = 1

Base.getindex(bt::AbstractBlockTuple, i::Integer) = Tuple(bt)[i]
Base.getindex(bt::AbstractBlockTuple, r::AbstractUnitRange) = Tuple(bt)[r]
Base.getindex(bt::AbstractBlockTuple, b::Block{1}) = blocks(bt)[Int(b)]
function Base.getindex(bt::AbstractBlockTuple, br::BlockRange{1})
  r = Int.(br)
  T = unspecify_type_parameters(typeof(bt))
  flat = Tuple(bt)[blockfirsts(bt)[first(r)]:blocklasts(bt)[last(r)]]
  return T{blocklengths(bt)[r]}(flat)
end
function Base.getindex(bt::AbstractBlockTuple, bi::BlockIndexRange{1})
  return bt[Block(bi)][only(bi.indices)]
end

Base.iterate(bt::AbstractBlockTuple) = iterate(Tuple(bt))
Base.iterate(bt::AbstractBlockTuple, i::Int) = iterate(Tuple(bt), i)

Base.length(bt::AbstractBlockTuple) = length(Tuple(bt))

Base.lastindex(bt::AbstractBlockTuple) = length(bt)

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
  return BT{BlockLengths}(bc.f.((Tuple.(bc.args))...))
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
tuplemortar(tt::Vararg{Tuple}) = BlockedTuple{length.(tt)}(flatten_tuples(tt))
function BlockedTuple(flat::Tuple, BlockLengths::Tuple{Vararg{Int}})
  return BlockedTuple{BlockLengths}(flat)
end
BlockedTuple(bt::AbstractBlockTuple) = BlockedTuple{blocklengths(bt)}(Tuple(bt))

# Base interface
Base.Tuple(bt::BlockedTuple) = bt.flat

Base.map(f, bt::BlockedTuple) = BlockedTuple{blocklengths(bt)}(map(f, Tuple(bt)))

# BlockArrays interface
function BlockArrays.blocklengths(::Type{<:BlockedTuple{BlockLengths}}) where {BlockLengths}
  return BlockLengths
end
