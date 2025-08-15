using .BaseExtensions: BaseExtensions
using BlockArrays: blocklengths

# default: if no bipartion is specified, all axes to domain
invbiperm(perm, ::Any) = invbiperm(perm, Val(0))
invbiperm(perm, t::Tuple{Tuple,Tuple}) = invbiperm(perm, tuplemortar(t))
invbiperm(perm, t::AbstractBlockTuple{2}) = invbiperm(perm, Val(first(blocklength(t))))

function invbiperm(perm, ::Val{N1}) where {N1}
  perm_out = invperm(Tuple(perm))
  length(perm) <= N1 && return blockedpermvcat(perm_out, ())
  return blockedpermvcat(perm_out[begin:N1], (perm_out[(N1 + 1):end]))
end

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
  biperm_dest_to_a12 = (perm_codomain_dest..., perm_domain_dest...)
  biperm_a12_to_dest = invbiperm(biperm_dest_to_a12, dimnames_dest)

  perm_codomain1 = BaseExtensions.indexin(codomain, dimnames1)
  perm_domain1 = BaseExtensions.indexin(contracted, dimnames1)

  perm_codomain2 = BaseExtensions.indexin(contracted, dimnames2)
  perm_domain2 = BaseExtensions.indexin(domain, dimnames2)

  permblocks1 = (perm_codomain1, perm_domain1)
  biperm1 = blockedpermvcat(permblocks1...)
  permblocks2 = (perm_codomain2, perm_domain2)
  biperm2 = blockedpermvcat(permblocks2...)
  return biperm_a12_to_dest, biperm1, biperm2
end
