# `Base.indexin` doesn't accept tuples; return the positions of `x` in `y` as a tuple.
function tuple_indexin(x::Tuple, y::AbstractArray)
    return Tuple{Vararg{Any, length(x)}}(Base.indexin(x, y))
end
tuple_indexin(x::Tuple, y) = tuple_indexin(x, collect(y))

# Locate two subgroups `sub1`, `sub2` within `collection`, returning their two index groups.
function biindexin(collection, sub1, sub2)
    return tuple_indexin(sub1, collection), tuple_indexin(sub2, collection)
end

# Split `perm` into a codomain block of length `blocklength1` and a domain block.
function biperm(perm, blocklength1::Integer)
    return biperm(perm, Val(blocklength1))
end
function biperm(perm, ::Val{BlockLength1}) where {BlockLength1}
    length(perm) < BlockLength1 && throw(ArgumentError("Invalid codomain length"))
    return BiTuple(Tuple(perm), Val(BlockLength1))
end

length_domain(t::BiTuple) = length(t.t2)
# Assume all dimensions are in the codomain by default
length_domain(t) = 0

length_codomain(t) = length(t) - length_domain(t)

# codomain <-- domain
function blockedperms(::typeof(contract), dimnames_dest, dimnames1, dimnames2)
    dimnames = collect(Iterators.flatten((dimnames_dest, dimnames1, dimnames2)))
    for i in unique(dimnames)
        count(==(i), dimnames) == 2 || throw(ArgumentError("Invalid contraction labels"))
    end

    codomain = Tuple(setdiff(dimnames1, dimnames2))
    contracted = Tuple(intersect(dimnames1, dimnames2))
    domain = Tuple(setdiff(dimnames2, dimnames1))

    perm_codomain_dest = tuple_indexin(codomain, dimnames_dest)
    perm_domain_dest = tuple_indexin(domain, dimnames_dest)
    invbiperm = (perm_codomain_dest..., perm_domain_dest...)
    biperm_dest = biperm(invperm(invbiperm), length(codomain))

    perm_codomain1 = tuple_indexin(codomain, dimnames1)
    perm_domain1 = tuple_indexin(contracted, dimnames1)

    perm_codomain2 = tuple_indexin(contracted, dimnames2)
    perm_domain2 = tuple_indexin(domain, dimnames2)

    biperm1 = BiTuple(perm_codomain1, perm_domain1)
    biperm2 = BiTuple(perm_codomain2, perm_domain2)
    return biperm_dest, biperm1, biperm2
end
