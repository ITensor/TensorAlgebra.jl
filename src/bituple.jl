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
