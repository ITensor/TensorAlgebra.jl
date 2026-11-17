import TupleTools

# `a ∖ b` as a `Vector`, preserving the order of `a`. A linear scan is faster than `Base.setdiff`
# for the small collections here.
smallsetdiff(a, b) = [x for x in a if x ∉ b]

# Position of each element of `x` in `y`, as a tuple. Linear scan, no hashing
# (`Base.indexin` builds a `Dict`), for the small collections here.
tuple_indexin(x::Tuple, y) = map(v -> findfirst(==(v), y), x)

"""
    biperm(t, t1, t2) -> (p1, p2)

Locate the groups `t1` and `t2` within `t`, returning the positions of `t1` as
`p1` and the positions of `t2` as `p2`. The groups `t1` and `t2` must partition
`t`, so the concatenation `(p1..., p2...)` is a permutation of `eachindex(t)` and
the pair `(p1, p2)` is a bipartitioned permutation (a "biperm") splitting `t`
into a codomain `p1` and a domain `p2`.
"""
function biperm(t, t1, t2)
    length(t1) + length(t2) == length(t) || throw(
        ArgumentError(
            "groups of lengths $(length(t1)) and $(length(t2)) do not partition a collection of length $(length(t))"
        )
    )
    return tuple_indexin(t1, t), tuple_indexin(t2, t)
end

length_domain(t::BiTuple) = length(t.t2)
# Assume all dimensions are in the codomain by default
length_domain(t) = 0

length_codomain(t) = length(t) - length_domain(t)

# `findfirst` for a match the caller guarantees exists, so the result is an `Int` rather
# than `Union{Int, Nothing}` (the `Nothing` would otherwise break inference downstream).
checked_findfirst(pred, collection) = something(findfirst(pred, collection))

# codomain <-- domain
function biperms(::typeof(contract), labels_dest, labels1, labels2)
    t1, t2 = Tuple(labels1), Tuple(labels2)
    contracted1 = map(in(t2), t1)
    return biperms(contract, Val(count(contracted1)), labels_dest, t1, t2, contracted1)
end
# `K`, the number of contracted labels, is a `Val` so the permutations are concretely typed
# tuples. `contracted1` is the mask of contracted labels in `labels1`.
function biperms(
        ::typeof(contract), ::Val{K}, labels_dest, labels1, labels2, contracted1
    ) where {K}
    n1, n2 = length(labels1), length(labels2)
    # `sortperm` of the boolean mask is a stable partition: uncontracted (`false`) indices
    # first, contracted (`true`) indices last, each in their original order.
    perm1_codomain, perm1_domain =
        bipartition(TupleTools.sortperm(contracted1), Val(n1 - K))
    perm2_domain, _ =
        bipartition(TupleTools.sortperm(map(in(labels1), labels2)), Val(n2 - K))
    # Align the contracted groups: list operand 2's contracted labels in operand 1's order.
    perm2_codomain = map(p -> checked_findfirst(==(labels1[p]), labels2), perm1_domain)
    # The operand groups are consistent by construction, so only the destination labels need
    # checking. `checked_findfirst` below catches any that are missing.
    length(labels_dest) == (n1 - K) + (n2 - K) ||
        throw(ArgumentError("Invalid contraction labels"))
    pos_dest = (
        map(p -> checked_findfirst(==(labels1[p]), labels_dest), perm1_codomain)...,
        map(p -> checked_findfirst(==(labels2[p]), labels_dest), perm2_domain)...,
    )
    biperm_dest = bipartition(invperm(pos_dest), Val(n1 - K))
    return biperm_dest, (perm1_codomain, perm1_domain), (perm2_codomain, perm2_domain)
end
