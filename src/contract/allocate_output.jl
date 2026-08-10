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
    check_biperm(a1, perm1_codomain, perm1_domain)
    check_biperm(a2, perm2_codomain, perm2_domain)
    # The contracted axes of `a1` (its domain) pair with the contracted axes of `a2` (its codomain),
    # and each pair must be a dual pair: `dual(ax1) == ax2`. `dual` falls back to the identity for
    # plain ranges, so this reduces to `ax1 == ax2` for non-graded arrays.
    length(perm1_domain) == length(perm2_codomain) || throw(
        ArgumentError(
            "Number of contracted axes do not match: `a1` has $(length(perm1_domain)), `a2` has $(length(perm2_codomain))"
        )
    )
    for (i, j) in zip(perm1_domain, perm2_codomain)
        ax1 = axes(a1, i)
        ax2 = axes(a2, j)
        dual(ax1) == ax2 || throw(
            ArgumentError(
                "Contracted axes do not match: `axes(a1, $i) = $ax1` and `axes(a2, $j) = $ax2`"
            )
        )
    end
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
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    axes_codomain, _ = bipartition(axes(a1), perm1_codomain, perm1_domain)
    _, axes_domain = bipartition(axes(a2), perm2_codomain, perm2_domain)
    axes_uncontracted = (axes_codomain..., axes_domain...)
    return bipartition(axes_uncontracted, perm_dest_codomain, perm_dest_domain)
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function allocate_output(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
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
    codomain_axes_dest, domain_axes_dest = output_axes(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    T = promote_type(eltype(a1), eltype(a2))
    # `domain_axes_dest` come straight from `axes(a2)` (stored/dualized convention), so
    # un-dualize them into `similar_map`'s codomain-facing convention.
    return zero!(similar_map(a1, T, codomain_axes_dest, conj.(domain_axes_dest)))
end
