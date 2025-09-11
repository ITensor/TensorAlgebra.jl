using .BaseExtensions: BaseExtensions
using BlockArrays: blocklengths

# default: if no bipartion is specified, all axes to domain
function biperm(perm, blocklength1::Integer)
  return biperm(perm, Val(blocklength1))
end
function biperm(perm, ::Val{BlockLength1}) where {BlockLength1}
  length(perm) < BlockLength1 && throw(ArgumentError("Invalid codomain length"))
  return blockedperm(Tuple(perm), (BlockLength1, length(perm) - BlockLength1))
end

length_domain(t::AbstractBlockTuple{2}) = last(blocklengths(t))
# Assume all dimensions are in the codomain by default
length_domain(t) = 0

length_codomain(t) = length(t) - length_domain(t)

function blockedperms(
  f::typeof(contract), alg::Algorithm, dimnames_dest, dimnames1, dimnames2
)
  return blockedperms(f, dimnames_dest, dimnames1, dimnames2)
end

# codomain <-- domain
function blockedperms(::typeof(contract), dimnames_dest, dimnames1, dimnames2)
  dimnames = collect(Iterators.flatten((dimnames_dest, dimnames1, dimnames2)))
  for i in unique(dimnames)
    count(==(i), dimnames) == 2 || throw(ArgumentError("Invalid contraction labels"))
  end

  codomain = Tuple(setdiff(dimnames1, dimnames2))
  contracted = Tuple(intersect(dimnames1, dimnames2))
  domain = Tuple(setdiff(dimnames2, dimnames1))

  perm_codomain_dest = BaseExtensions.indexin(codomain, dimnames_dest)
  perm_domain_dest = BaseExtensions.indexin(domain, dimnames_dest)
  invbiperm = (perm_codomain_dest..., perm_domain_dest...)
  biperm_dest = biperm(invperm(invbiperm), length_codomain(dimnames_dest))

  perm_codomain1 = BaseExtensions.indexin(codomain, dimnames1)
  perm_domain1 = BaseExtensions.indexin(contracted, dimnames1)

  perm_codomain2 = BaseExtensions.indexin(contracted, dimnames2)
  perm_domain2 = BaseExtensions.indexin(domain, dimnames2)

  permblocks1 = (perm_codomain1, perm_domain1)
  biperm1 = blockedpermvcat(permblocks1...)
  permblocks2 = (perm_codomain2, perm_domain2)
  biperm2 = blockedpermvcat(permblocks2...)
  return biperm_dest, biperm1, biperm2
end
