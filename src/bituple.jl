# A two-block tuple: a flat tuple split into a codomain group `t1` and a domain group `t2`.
# This is the only blocked-tuple shape matricization, contraction, and factorizations ever use.
# When its entries are `Int`s forming a permutation it acts as a "biperm"; the permutation
# property is validated at construction by the perm builders (like `permutedims(a, perm)`),
# not encoded in a separate type.

unval(::Val{N}) where {N} = N

struct BiTuple{N1, N2, T1 <: NTuple{N1, Any}, T2 <: NTuple{N2, Any}}
    t1::T1
    t2::T2
end

# Accessors (a small BlockArrays-inspired surface).
blocks(bt::BiTuple) = (bt.t1, bt.t2)
firstblock(bt::BiTuple) = bt.t1
lastblock(bt::BiTuple) = bt.t2
blocklengths(::BiTuple{N1, N2}) where {N1, N2} = (N1, N2)
blocklength(::BiTuple) = 2

Base.Tuple(bt::BiTuple) = (bt.t1..., bt.t2...)
Base.length(::BiTuple{N1, N2}) where {N1, N2} = N1 + N2
Base.iterate(bt::BiTuple, args...) = iterate(Tuple(bt), args...)
Base.getindex(bt::BiTuple, i::Integer) = Tuple(bt)[i]
function Base.eltype(::Type{<:BiTuple{<:Any, <:Any, T1, T2}}) where {T1, T2}
    return promote_type(eltype(T1), eltype(T2))
end

function Base.show(io::IO, bt::BiTuple)
    return print(io, "tuplemortar(", blocks(bt), ")")
end

function Base.invperm(bt::BiTuple{N1, N2}) where {N1, N2}
    ip = invperm(Tuple(bt))
    return BiTuple(ntuple(i -> ip[i], Val(N1)), ntuple(i -> ip[N1 + i], Val(N2)))
end

#
# Constructors
#

# Axis bituple: split a tuple-of-tuples into the two groups verbatim.
tuplemortar(tt::Tuple{Tuple, Tuple}) = BiTuple(tt[1], tt[2])

# Permutation bituples: validate at runtime that the flat tuple is a permutation.
function permmortar(permblocks::Tuple{Tuple{Vararg{Int}}, Tuple{Vararg{Int}}})
    bt = BiTuple(permblocks[1], permblocks[2])
    @assert isperm(Tuple(bt))
    return bt
end

# Split a flat permutation into a codomain block of length `blocklengths[1]` and the rest.
function blockedperm(perm::Tuple{Vararg{Int}}, blocklengths::Tuple{Int, Int})
    l1 = blocklengths[1]
    return permmortar((perm[1:l1], perm[(l1 + 1):end]))
end

# Two-block vcat builder (the contraction path's entry point).
function blockedpermvcat(block1::Tuple{Vararg{Int}}, block2::Tuple{Vararg{Int}})
    return permmortar((block1, block2))
end

# `indexin`-based builder: locate each group's labels within `collection`.
function blockedperm_indexin(collection, sub1, sub2)
    return permmortar(
        (
            BaseExtensions.indexin(sub1, collection),
            BaseExtensions.indexin(sub2, collection),
        )
    )
end

# Trivial (identity) biperm with the codomain/domain split given by `Val`s. Known to be a
# permutation by construction, so it skips the `isperm` check on this hot path.
function trivialbiperm(::Val{N1}, ::Val{N}) where {N1, N}
    return BiTuple(ntuple(identity, Val(N1)), ntuple(i -> N1 + i, Val(N - N1)))
end

# Bipartition a collection according to a biperm (out-of-place, blocked).
function blockpermute(v, bt::BiTuple)
    return BiTuple(map(i -> v[i], bt.t1), map(i -> v[i], bt.t2))
end
