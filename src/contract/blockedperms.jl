using .BaseExtensions: BaseExtensions

function blockedperms(
  f::typeof(contract), alg::Algorithm, dimnames_dest, dimnames1, dimnames2
)
  return blockedperms(f, dimnames_dest, dimnames1, dimnames2)
end

# codomain <-- domain
function blockedperms(::typeof(contract), dimnames_dest, dimnames1, dimnames2)
  codomain = Tuple(setdiff(dimnames1, dimnames2))
  contracted = Tuple(intersect(dimnames1, dimnames2))
  domain = Tuple(setdiff(dimnames2, dimnames1))

  perm_codomain_dest = BaseExtensions.indexin(codomain, dimnames_dest)
  perm_domain_dest = BaseExtensions.indexin(domain, dimnames_dest)

  perm_codomain1 = BaseExtensions.indexin(codomain, dimnames1)
  perm_domain1 = BaseExtensions.indexin(contracted, dimnames1)

  perm_codomain2 = BaseExtensions.indexin(contracted, dimnames2)
  perm_domain2 = BaseExtensions.indexin(domain, dimnames2)

  permblocks_dest = (perm_codomain_dest, perm_domain_dest)
  biperm_dest = blockedpermvcat(permblocks_dest...)
  permblocks1 = (perm_codomain1, perm_domain1)
  biperm1 = blockedpermvcat(permblocks1...)
  permblocks2 = (perm_codomain2, perm_domain2)
  biperm2 = blockedpermvcat(permblocks2...)
  return biperm_dest, biperm1, biperm2
end
