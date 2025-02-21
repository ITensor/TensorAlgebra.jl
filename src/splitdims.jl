using .BaseExtensions: _permutedims, _permutedims!

to_axis(a::AbstractUnitRange) = a
to_axis(n::Integer) = Base.OneTo(n)

function blockedaxes(a::AbstractArray, sizeblocks::Pair...)
  axes_split = tuple.(axes(a))
  for (dim, sizeblock) in sizeblocks
    # TODO: Handle conversion from length to range!
    axes_split = Base.setindex(axes_split, to_axis.(sizeblock), dim)
  end
  return tuplemortar(axes_split)
end

function splitdims(::ReshapeFusion, a::AbstractArray, abt::BlockedTuple)
  # TODO: Add `uncanonicalizedims`.
  # TODO: Need `length` since `reshape` doesn't accept `axes`,
  # maybe make a `reshape_axes` function.
  return reshape(a, Tuple(length.(abt)))
end

# ambiguity for zero-dim
function splitdims(a::AbstractArray{<:Any,N}, abt::BlockedTuple{N,<:Any,Tuple{}}) where {N}
  return splitdims(FusionStyle(a), a, abt)
end

function splitdims(
  a::AbstractArray{<:Any,N}, bt::BlockedTuple{N,<:Any,<:Tuple{Vararg{AbstractUnitRange}}}
) where {N}
  return splitdims(FusionStyle(a), a, bt)
end

# splitdims(randn(4, 4), 1:2, 1:2, 1:2, 1:2)
function splitdims(a::AbstractArray, axes::AbstractUnitRange...)
  return splitdims(a, tuple.(axes)...)
end

# splitdims(randn(4, 4), (1:2, 1:2), (1:2, 1:2))
function splitdims(a::AbstractArray, axesblocks::Tuple{Vararg{AbstractUnitRange}}...)
  # TODO: Add `uncanonicalizedims`.
  return splitdims(a, tuplemortar(axesblocks))
end

# Fix ambiguity issue
splitdims(a::AbstractArray) = a

# splitdims(randn(4, 4), (2, 2), (2, 2))
function splitdims(a::AbstractArray, sizeblocks::Tuple{Vararg{Integer}}...)
  return splitdims(a, tuplemortar(sizeblocks))
end

# splitdims(randn(4, 4), tuplemortar(((2, 2), (2, 2))))
function splitdims(
  a::AbstractArray{<:Any,N}, bt::BlockedTuple{N,<:Any,<:Tuple{Vararg{Integer}}}
) where {N}
  return splitdims(a, to_axis.(bt))
end

# splitdims(randn(4, 4), 2 => (1:2, 1:2))
function splitdims(a::AbstractArray, sizeblocks::Pair...)
  return splitdims(a, blockedaxes(a, sizeblocks...))
end

function splitdims!(
  a_dest::AbstractArray, a::AbstractArray, blockedperm::AbstractBlockPermutation
)
  axes_dest = map(i -> axes(a_dest, i), blockedperm)
  a_dest_perm = splitdims(a, axes_dest)
  _permutedims!(a_dest, a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end
