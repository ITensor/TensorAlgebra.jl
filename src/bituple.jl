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
Base.hash(bt::BiTuple, h::UInt) = hash(bt.t2, hash(bt.t1, hash(:BiTuple, h)))

Base.invperm(bt::BiTuple{N1}) where {N1} = BiTuple(invperm(Tuple(bt)), Val(N1))

# Partition `v` into two groups. The partition is specified either by a split length (take
# the first `length1` entries in order, then the rest), by two index groups `t1`/`t2`, or by
# a `BiTuple` of index groups.
function bipartition(t::Tuple, length1::Val)
    bt = BiTuple(t, length1)
    return bt.t1, bt.t2
end
bipartition(v, t1::Tuple, t2::Tuple) = (map(i -> v[i], t1), map(i -> v[i], t2))
bipartition(v, bt::BiTuple) = bipartition(v, bt.t1, bt.t2)
