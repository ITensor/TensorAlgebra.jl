# This file defines BlockedTuple, a Tuple of heterogeneous Tuple with a BlockArrays.jl
# like interface

using BlockArrays: Block, BlockArrays, BlockIndexRange, BlockRange, blockedrange

struct BlockedTuple{Divs,Flat}
  flat::Flat

  function BlockedTuple{Divs}(flat::Tuple) where {Divs}
    length(flat) != sum(Divs) && throw(DimensionMismatch("Invalid total length"))
    return new{Divs,typeof(flat)}(flat)
  end
end

# TensorAlgebra Interface
BlockedTuple(tt::Vararg{Tuple}) = BlockedTuple{length.(tt)}(flatten_tuples(tt))
BlockedTuple(bt::BlockedTuple) = bt
flatten_tuples(bt::BlockedTuple) = Tuple(bt)

# Base interface
Base.Tuple(bt::BlockedTuple) = bt.flat

Base.axes(bt::BlockedTuple) = (blockedrange([blocklengths(bt)...]),)

Base.broadcastable(bt::BlockedTuple) = bt
struct BlockedTupleBroadcastStyle{Divs} <: Broadcast.BroadcastStyle end
function Base.BroadcastStyle(::Type{<:BlockedTuple{Divs}}) where {Divs}
  return BlockedTupleBroadcastStyle{Divs}()
end
function Base.BroadcastStyle(::BlockedTupleBroadcastStyle, ::BlockedTupleBroadcastStyle)
  throw(DimensionMismatch("Incompatible blocks"))
end
# BroadcastStyle is not called for two identical styles
function Base.copy(bc::Broadcast.Broadcasted{BlockedTupleBroadcastStyle{Divs}}) where {Divs}
  return BlockedTuple{Divs}(bc.f.((Tuple.(bc.args))...))
end

Base.copy(bt::BlockedTuple) = BlockedTuple{blocklengths(bt)}(copy.(Tuple(bt)))

Base.deepcopy(bt::BlockedTuple) = BlockedTuple{blocklengths(bt)}(deepcopy.(Tuple(bt)))

Base.firstindex(::BlockedTuple) = 1

Base.getindex(bt::BlockedTuple, i::Integer) = Tuple(bt)[i]
Base.getindex(bt::BlockedTuple, r::AbstractUnitRange) = Tuple(bt)[r]
Base.getindex(bt::BlockedTuple, b::Block{1}) = blocks(bt)[Int(b)]
Base.getindex(bt::BlockedTuple, br::BlockRange{1}) = blocks(bt)[Int.(br)]
Base.getindex(bt::BlockedTuple, bi::BlockIndexRange{1}) = bt[Block(bi)][only(bi.indices)]

Base.iterate(bt::BlockedTuple) = iterate(Tuple(bt))
Base.iterate(bt::BlockedTuple, i::Int) = iterate(Tuple(bt), i)

Base.lastindex(bt::BlockedTuple) = length(bt)

Base.length(bt::BlockedTuple) = length(Tuple(bt))

Base.map(f, bt::BlockedTuple) = BlockedTuple{blocklengths(bt)}(map(f, Tuple(bt)))

# BlockArrays interface
function BlockArrays.blockfirsts(bt::BlockedTuple)
  return (0, cumsum(blocklengths(bt)[begin:(end - 1)])...) .+ 1
end

function BlockArrays.blocklasts(bt::BlockedTuple)
  return cumsum(blocklengths(bt)[begin:end])
end

BlockArrays.blocklength(::BlockedTuple{Divs}) where {Divs} = length(Divs)

BlockArrays.blocklengths(::BlockedTuple{Divs}) where {Divs} = Divs

function BlockArrays.blocks(bt::BlockedTuple)
  bf = blockfirsts(bt)
  bl = blocklasts(bt)
  return ntuple(i -> Tuple(bt)[bf[i]:bl[i]], blocklength(bt))
end
