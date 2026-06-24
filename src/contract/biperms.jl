# `Base.indexin` doesn't accept tuples; return the positions of `x` in `y` as a tuple.
function tuple_indexin(x::Tuple, y::AbstractArray)
    return Tuple{Vararg{Any, length(x)}}(Base.indexin(x, y))
end
tuple_indexin(x::Tuple, y) = tuple_indexin(x, collect(y))

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

# codomain <-- domain
function biperms(::typeof(contract), dimnames_dest, dimnames1, dimnames2)
    dimnames = collect(Iterators.flatten((dimnames_dest, dimnames1, dimnames2)))
    for i in unique(dimnames)
        count(==(i), dimnames) == 2 || throw(ArgumentError("Invalid contraction labels"))
    end

    codomain = Tuple(setdiff(dimnames1, dimnames2))
    contracted = Tuple(intersect(dimnames1, dimnames2))
    domain = Tuple(setdiff(dimnames2, dimnames1))

    perm_codomain_dest, perm_domain_dest = biperm(dimnames_dest, codomain, domain)
    invperm_dest = invperm((perm_codomain_dest..., perm_domain_dest...))
    biperm_dest = bipartition(invperm_dest, Val(length(codomain)))

    biperm1 = biperm(dimnames1, codomain, contracted)
    biperm2 = biperm(dimnames2, contracted, domain)
    return biperm_dest, biperm1, biperm2
end
