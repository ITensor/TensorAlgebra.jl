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
    axes_codomain_dest, axes_domain_dest = bipartition(
        axes_uncontracted, perm_dest_codomain, perm_dest_domain
    )
    # The operand axes are stored/dualized, so un-dualize the domain axes into the codomain-facing
    # construction convention shared by `allocate_contract_output`, `similar_map`, and `unmatricize`
    # (a no-op on dense axes).
    return axes_codomain_dest, conj.(axes_domain_dest)
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
    axes_codomain_dest, axes_domain_dest = output_axes(
        contract,
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return allocate_contract_output(a1, a2, T, axes_codomain_dest, axes_domain_dest)
end

# Allocate the output container for `contract`: the operand types, the output element type and
# axes (domain codomain-facing), and the output's codomain/domain leg counts (the axes tuple
# lengths) select the container type. Internal to TensorAlgebra, not a public extension point:
# the leg counts identify the contraction pattern only for matrix-shaped operands (see the
# `Diagonal` method in `diagonal.jl`), so external structured types should not overload it.
function allocate_contract_output(a1, a2, T, axes_codomain::Tuple, axes_domain::Tuple)
    return zero!(similar_map(a1, T, axes_codomain, axes_domain))
end
