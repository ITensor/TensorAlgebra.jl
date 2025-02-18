using BlockArrays:
  BlockArrays, Block, blockfirsts, blocklasts, blocklength, blocklengths, blocks
using EllipsisNotation: Ellipsis, var".."
using TupleTools: TupleTools

trivialperm(len) = ntuple(identity, len)
function istrivialperm(t::Tuple)
  return t == trivialperm(length(t))
end

value(::Val{N}) where {N} = N

_flatten_tuples(t::Tuple) = t
function _flatten_tuples(t1::Tuple, t2::Tuple, trest::Tuple...)
  return _flatten_tuples((t1..., t2...), trest...)
end
_flatten_tuples() = ()
flatten_tuples(ts::Tuple) = _flatten_tuples(ts...)

collect_tuple(x) = (x,)
collect_tuple(x::Ellipsis) = x
collect_tuple(t::Tuple) = t

#
# ===============================  AbstractBlockPermutation  ===============================
#
abstract type AbstractBlockPermutation{BlockLength} <: AbstractBlockTuple{BlockLength} end

widened_constructorof(::Type{<:AbstractBlockPermutation}) = BlockedTuple

# Block a permutation based on the specified lengths.
# blockperm((4, 3, 2, 1), (2, 2)) == blockedperm((4, 3), (2, 1))
# TODO: Optimize with StaticNumbers.jl or generated functions, see:
# https://discourse.julialang.org/t/avoiding-type-instability-when-slicing-a-tuple/38567
function blockperm(perm::Tuple{Vararg{Int}}, blocklengths::Tuple{Vararg{Int}})
  return blockedperm(BlockedTuple(perm, blocklengths))
end

function blockperm(perm::Tuple{Vararg{Int}}, BlockLengths::Val)
  return blockedperm(BlockedTuple(perm, BlockLengths))
end

function Base.invperm(blockedperm::AbstractBlockPermutation)
  # use Val to preserve compile time info
  return blockperm(invperm(Tuple(blockedperm)), Val(blocklengths(blockedperm)))
end

#
# Constructors
#

# Bipartition a vector according to the
# bipartitioned permutation.
# Like `Base.permute!` block out-of-place and blocked.
function blockpermute(v, blockedperm::AbstractBlockPermutation)
  return map(blockperm -> map(i -> v[i], blockperm), blocks(blockedperm))
end

# blockedperm((4, 3), (2, 1))
function blockedperm(permblocks::Tuple{Vararg{Int}}...; length::Union{Val,Nothing}=nothing)
  return blockedperm(length, permblocks...)
end

function blockedperm(::Nothing, permblocks::Tuple{Vararg{Int}}...)
  return blockedperm(Val(sum(length, permblocks; init=zero(Bool))), permblocks...)
end

# blockedperm((3, 2), 1) == blockedperm((3, 2), (1,))
function blockedperm(permblocks::Union{Tuple{Vararg{Int}},Int}...; kwargs...)
  return blockedperm(collect_tuple.(permblocks)...; kwargs...)
end

function blockedperm(
  permblocks::Union{Tuple{Vararg{Int}},Tuple{Ellipsis},Int,Ellipsis}...; kwargs...
)
  return blockedperm(collect_tuple.(permblocks)...; kwargs...)
end

function blockedperm(bt::AbstractBlockTuple)
  return blockedperm(Val(length(bt)), blocks(bt)...)
end

function _blockedperm_length(::Nothing, specified_perm::Tuple{Vararg{Int}})
  return maximum(specified_perm)
end

function _blockedperm_length(vallength::Val, ::Tuple{Vararg{Int}})
  return value(vallength)
end

# blockedperm((4, 3), .., 1) == blockedperm((4, 3), (2,), (1,))
# blockedperm((4, 3), .., 1; length=Val(5)) == blockedperm((4, 3), (2,), (5,), (1,))
# blockedperm((4, 3), (..,), 1) == blockedperm((4, 3), (2,), (1,))
# blockedperm((4, 3), (..,), 1; length=Val(5)) == blockedperm((4, 3), (2, 5), (1,))
function blockedperm(
  permblocks::Union{Tuple{Vararg{Int}},Ellipsis,Tuple{Ellipsis}}...;
  length::Union{Val,Nothing}=nothing,
)
  # Check there is only one `Ellipsis`.
  @assert isone(count(x -> x isa Union{Ellipsis,Tuple{Ellipsis}}, permblocks))
  specified_permblocks = filter(x -> !(x isa Union{Ellipsis,Tuple{Ellipsis}}), permblocks)
  unspecified_dim = findfirst(x -> x isa Union{Ellipsis,Tuple{Ellipsis}}, permblocks)
  specified_perm = flatten_tuples(specified_permblocks)
  len = _blockedperm_length(length, specified_perm)
  unspecified_dims_vec = setdiff(Base.OneTo(len), specified_perm)
  UD = Val(len - sum(Base.length.(specified_permblocks)))  # preserve type stability when possible
  insert = unspecified_dims(typeof(permblocks[unspecified_dim]), unspecified_dims_vec, UD)
  permblocks_specified = TupleTools.insertat(permblocks, unspecified_dim, insert)
  return blockedperm(permblocks_specified...)
end

function unspecified_dims(::Type{Tuple{Ellipsis}}, unspecified_dims_vec, UD::Val)
  return (NTuple{value(UD),Int}(unspecified_dims_vec),)
end
function unspecified_dims(::Type{Ellipsis}, unspecified_dims_vec, UD::Val)
  return NTuple{value(UD),Tuple{Int}}(Tuple.(unspecified_dims_vec))
end

# Version of `indexin` that outputs a `blockedperm`.
function blockedperm_indexin(collection, subs...)
  return blockedperm(map(sub -> BaseExtensions.indexin(sub, collection), subs)...)
end

#
# ==================================  BlockedPermutation  ==================================
#

# for dispatch reason, it is convenient to have BlockLength as the first parameter
struct BlockedPermutation{BlockLength,BlockLengths,Flat} <:
       AbstractBlockPermutation{BlockLength}
  flat::Flat

  function BlockedPermutation{BlockLength,BlockLengths}(
    flat::Tuple
  ) where {BlockLength,BlockLengths}
    length(flat) != sum(BlockLengths; init=0) &&
      throw(DimensionMismatch("Invalid total length"))
    length(BlockLengths) != BlockLength &&
      throw(DimensionMismatch("Invalid total blocklength"))
    any(BlockLengths .< 0) && throw(DimensionMismatch("Invalid block length"))
    return new{BlockLength,BlockLengths,typeof(flat)}(flat)
  end
end

# Base interface
Base.Tuple(blockedperm::BlockedPermutation) = getfield(blockedperm, :flat)

# BlockArrays interface
function BlockArrays.blocklengths(
  ::Type{<:BlockedPermutation{<:Any,BlockLengths}}
) where {BlockLengths}
  return BlockLengths
end

function blockedperm(::Val, permblocks::Tuple{Vararg{Int}}...)
  blockedperm = BlockedPermutation{length(permblocks),length.(permblocks)}(
    flatten_tuples(permblocks)
  )
  @assert isperm(blockedperm)
  return blockedperm
end

#
# ==============================  BlockedTrivialPermutation  ===============================
#
trivialperm(length::Union{Integer,Val}) = ntuple(identity, length)

struct BlockedTrivialPermutation{BlockLength,BlockLengths} <:
       AbstractBlockPermutation{BlockLength} end

Base.Tuple(blockedperm::BlockedTrivialPermutation) = trivialperm(length(blockedperm))

# BlockArrays interface
function BlockArrays.blocklengths(
  ::Type{<:BlockedTrivialPermutation{<:Any,BlockLengths}}
) where {BlockLengths}
  return BlockLengths
end

blockedperm(tp::BlockedTrivialPermutation) = tp

function blockedtrivialperm(blocklengths::Tuple{Vararg{Int}})
  return BlockedTrivialPermutation{length(blocklengths),blocklengths}()
end

function trivialperm(blockedperm::AbstractBlockTuple)
  return blockedtrivialperm(blocklengths(blockedperm))
end
Base.invperm(blockedperm::BlockedTrivialPermutation) = blockedperm
