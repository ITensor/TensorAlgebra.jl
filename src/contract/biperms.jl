import TupleTools

# `a ∖ b` as a `Vector`, preserving the order of `a`, via a linear scan. For the small
# collections here `Base.setdiff` is slower because it builds a `Set` and hashes; it
# assumes set-like (unique) inputs. Used to assemble the destination labels in
# `contract_labels`.
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

# Position of `x` in `labels`, assuming it is present (the caller guarantees this), so the
# result is an `Int` rather than `Union{Int, Nothing}`.
findpos(x, labels) = something(findfirst(==(x), labels))

# codomain <-- domain
function biperms(::typeof(contract), labels_dest, labels1, labels2)
    t1, t2 = Tuple(labels1), Tuple(labels2)
    contracted1 = map(in(t2), t1)
    return biperms(contract, Val(count(contracted1)), labels_dest, t1, t2, contracted1)
end
# `K` is the number of contracted labels. Passing it as a `Val` makes the group sizes
# compile-time constants, so the permutations below come out as concretely-typed tuples and
# the rest of the contraction stays type-stable. `contracted1` is the boolean mask of which
# of `labels1`'s labels are contracted (its `count` is `K`), threaded in from the caller.
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
    perm2_codomain = map(p -> findpos(labels1[p], labels2), perm1_domain)
    # The operands partition into (un)contracted groups by construction; the only label
    # consistency left to check is that the destination carries exactly the uncontracted
    # labels. Locating each below then checks they all land in the destination.
    length(labels_dest) == (n1 - K) + (n2 - K) ||
        throw(ArgumentError("Invalid contraction labels"))
    pos_dest = (
        map(p -> findpos(labels1[p], labels_dest), perm1_codomain)...,
        map(p -> findpos(labels2[p], labels_dest), perm2_domain)...,
    )
    biperm_dest = bipartition(invperm(pos_dest), Val(n1 - K))
    return biperm_dest, (perm1_codomain, perm1_domain), (perm2_codomain, perm2_domain)
end
