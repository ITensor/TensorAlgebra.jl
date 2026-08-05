# A bipartitioned tuple: a flat tuple carrying an extra codomain/domain split. It acts like the
# flat tuple `(t1..., t2...)` for iteration, indexing, and `length`, with the split exposed only
# through the `t1` and `t2` fields (the way `Pair` exposes its two halves through fields rather
# than a collection interface). When its entries are `Int`s forming a permutation it acts as a
# "biperm"; whether it is a valid permutation is the concern of the operation using it as one
# (e.g. `matricize`), not of the type.

unval(::Val{N}) where {N} = N

struct BiTuple{N1, N2, T1 <: NTuple{N1, Any}, T2 <: NTuple{N2, Any}}
    t1::T1
    t2::T2
end

# Split a flat tuple into a first group of length `N1` and the remaining second group.
function BiTuple(t::NTuple{N, Any}, ::Val{N1}) where {N, N1}
    return BiTuple(ntuple(i -> t[i], Val(N1)), ntuple(i -> t[N1 + i], Val(N - N1)))
end

Base.Tuple(bt::BiTuple) = (bt.t1..., bt.t2...)
Base.length(::BiTuple{N1, N2}) where {N1, N2} = N1 + N2
Base.iterate(bt::BiTuple, args...) = iterate(Tuple(bt), args...)
Base.getindex(bt::BiTuple, i::Integer) = Tuple(bt)[i]
function Base.eltype(::Type{<:BiTuple{<:Any, <:Any, T1, T2}}) where {T1, T2}
    return promote_type(eltype(T1), eltype(T2))
end

function Base.show(io::IO, bt::BiTuple)
    return print(io, "BiTuple(", bt.t1, ", ", bt.t2, ")")
end

Base.:(==)(a::BiTuple, b::BiTuple) = a.t1 == b.t1 && a.t2 == b.t2
# Compare flat against a plain tuple, so `axes(a) == (r1, r2)` works when `axes` returns a `BiTuple`.
Base.:(==)(bt::BiTuple, t::Tuple) = Tuple(bt) == t
Base.:(==)(t::Tuple, bt::BiTuple) = t == Tuple(bt)
Base.hash(bt::BiTuple, h::UInt) = hash(bt.t2, hash(bt.t1, hash(:BiTuple, h)))

Base.invperm(bt::BiTuple{N1}) where {N1} = BiTuple(invperm(Tuple(bt)), Val(N1))

# An `AbstractArray` whose `axes` is a `BiTuple` reaches a range of generic machinery (Base indexing,
# Base broadcasting/shape, and the matricize/permute helpers here) that assumes a flat tuple of axes.
# Index sets, bounds, shapes, and permutations all ignore the codomain/domain split, so flatten the
# `BiTuple` first. This is the same collapse-to-flat behavior mixed operations fall back to.
Base.LinearIndices(bt::BiTuple) = LinearIndices(Tuple(bt))
Base.CartesianIndices(bt::BiTuple) = CartesianIndices(Tuple(bt))
function Base.checkbounds_indices(::Type{Bool}, bt::BiTuple, I::Tuple)
    return Base.checkbounds_indices(Bool, Tuple(bt), I)
end
Base.promote_shape(a::BiTuple, b::Tuple) = Base.promote_shape(Tuple(a), b)
Base.promote_shape(a::Tuple, b::BiTuple) = Base.promote_shape(a, Tuple(b))
Base.promote_shape(a::BiTuple, b::BiTuple) = Base.promote_shape(Tuple(a), Tuple(b))
function Base.Broadcast.broadcast_shape(a::BiTuple, shapes...)
    return Base.Broadcast.broadcast_shape(Tuple(a), shapes...)
end
function Base.Broadcast.broadcast_shape(a::Tuple, b::BiTuple, shapes...)
    return Base.Broadcast.broadcast_shape(a, Tuple(b), shapes...)
end
function Base.Broadcast.check_broadcast_shape(a::BiTuple, b::Tuple{})
    return Base.Broadcast.check_broadcast_shape(Tuple(a), b)
end
function Base.Broadcast.check_broadcast_shape(a::BiTuple, b::Tuple)
    return Base.Broadcast.check_broadcast_shape(Tuple(a), b)
end
function Base.Broadcast.check_broadcast_shape(a, b::BiTuple)
    return Base.Broadcast.check_broadcast_shape(a, Tuple(b))
end
function Base.Broadcast.check_broadcast_shape(a::BiTuple, b::BiTuple)
    return Base.Broadcast.check_broadcast_shape(Tuple(a), Tuple(b))
end
function Base._show_nonempty(
        io::IO,
        X::AbstractMatrix,
        prefix::String,
        drop::Bool,
        axs::BiTuple
    )
    return Base._show_nonempty(io, X, prefix, drop, Tuple(axs))
end
function Base.PermutedDimsArrays.genperm(bt::BiTuple, perm::NTuple{N, Int}) where {N}
    return Base.PermutedDimsArrays.genperm(Tuple(bt), perm)
end
Base.tail(bt::BiTuple) = Base.tail(Tuple(bt))
Base.front(bt::BiTuple) = Base.front(Tuple(bt))
Base.lastindex(bt::BiTuple) = Base.lastindex(Tuple(bt))
# A single-argument `map` preserves the split, mapping each block: with one operand there is no
# ambiguity about which split to keep. (The multi-argument case is the ambiguous one and is left to
# collapse to the flat tuple.)
Base.map(f, bt::BiTuple) = BiTuple(map(f, bt.t1), map(f, bt.t2))
Base.safe_tail(bt::BiTuple) = Base.safe_tail(Tuple(bt))
Base.rdims(out::Val{N}, bt::BiTuple) where {N} = Base.rdims(out, Tuple(bt))
Base.offset_if_vec(i::Integer, axs::BiTuple) = Base.offset_if_vec(i, Tuple(axs))
Base.Broadcast._axes(bc::Base.Broadcast.Broadcasted, axs::BiTuple) = Tuple(axs)
bipartition_axes(bt::BiTuple, split...) = bipartition_axes(Tuple(bt), split...)
bipartition(bt::BiTuple, length1::Val) = bipartition(Tuple(bt), length1)
function bipartition(bt::BiTuple, group1::Tuple, group2::Tuple)
    return bipartition(Tuple(bt), group1, group2)
end

"""
    bipartition(t::Tuple, length1::Val) -> (t1, t2)
    bipartition(t::Tuple, group1::Tuple, group2::Tuple) -> (p1, p2)

Split a flat tuple into two groups, returned as a pair of tuples.

The first form splits `t` in order, taking the first `length1` entries as `t1`
and the remaining entries as `t2`. The second form gathers the entries of `t` at
the two index groups `group1` and `group2`, returning `p1 = t[group1...]` and
`p2 = t[group2...]`.
"""
function bipartition(t::Tuple, length1::Val)
    bt = BiTuple(t, length1)
    return bt.t1, bt.t2
end
function bipartition(t::Tuple, group1::Tuple, group2::Tuple)
    return map(i -> t[i], group1), map(i -> t[i], group2)
end
# Split `t` by the two groups of a `BiTuple`.
bipartition(t::Tuple, bt::BiTuple) = bipartition(t, bt.t1, bt.t2)
