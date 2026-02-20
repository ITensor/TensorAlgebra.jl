using Base.PermutedDimsArrays: genperm

function check_biperm(a, perm_codomain, perm_domain)
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    isperm((perm_codomain..., perm_domain...)) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    return nothing
end

function check_input(
        ::typeof(contract),
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    # TODO: FIXME: Check that contracted axes match.
    check_biperm(a1, perm1_codomain, perm1_domain)
    check_biperm(a2, perm2_codomain, perm2_domain)
    return nothing
end

function check_input(
        ::typeof(contract!),
        a_dest, perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    # TODO: FIXME: Check that uncontracted axes match.
    check_input(
        contract,
        a1,
        perm1_codomain,
        perm1_domain,
        a2,
        perm2_codomain,
        perm2_domain
    )
    check_biperm(a_dest, perm_dest_codomain, perm_dest_domain)
    return nothing
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function output_axes(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain
    )
    biperm1 = permmortar((perm1_codomain, perm1_domain))
    biperm2 = permmortar((perm2_codomain, perm2_domain))
    biperm_dest = permmortar((perm_dest_codomain, perm_dest_domain))
    axes_codomain, axes_contracted = blocks(axes(a1)[biperm1])
    axes_contracted2, axes_domain = blocks(axes(a2)[biperm2])
    @assert length.(axes_contracted) == length.(axes_contracted2)
    # default: flatten biperm_out
    return genperm((axes_codomain..., axes_domain...), Tuple(biperm_dest))
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function allocate_output(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain
    )
    check_input(
        contract,
        a1,
        perm1_codomain,
        perm1_domain,
        a2,
        perm2_codomain,
        perm2_domain
    )
    axes_dest = output_axes(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    return similar(a1, promote_type(eltype(a1), eltype(a2)), axes_dest)
end
