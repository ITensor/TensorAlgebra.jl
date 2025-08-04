using .BaseExtensions: BaseExtensions

function blockedperms(f::typeof(contract), ::Algorithm, dimnames1, dimnames2)
  return blockedperms(f, dimnames1, dimnames2)
end

# codomain <-- domain
function blockedperms(::typeof(contract), dimnames1, dimnames2)
  dimnames = collect(Iterators.flatten((dimnames1, dimnames2)))
  for i in unique(dimnames)
    count(==(i), dimnames) in (1, 2) || throw(ArgumentError("Invalid contraction labels"))
  end

  codomain = Tuple(setdiff(dimnames1, dimnames2))
  contracted = Tuple(intersect(dimnames1, dimnames2))
  domain = Tuple(setdiff(dimnames2, dimnames1))

  perm_codomain1 = BaseExtensions.indexin(codomain, dimnames1)
  perm_domain1 = BaseExtensions.indexin(contracted, dimnames1)

  perm_codomain2 = BaseExtensions.indexin(contracted, dimnames2)
  perm_domain2 = BaseExtensions.indexin(domain, dimnames2)

  permblocks1 = (perm_codomain1, perm_domain1)
  biperm1 = blockedpermvcat(permblocks1...)
  permblocks2 = (perm_codomain2, perm_domain2)
  biperm2 = blockedpermvcat(permblocks2...)
  return biperm1, biperm2
end
