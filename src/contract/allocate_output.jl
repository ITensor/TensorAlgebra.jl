function check_biperm(a, perm_codomain, perm_domain)
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    isbiperm(perm_codomain, perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    return nothing
end

# The contracted axes of `op1(a1)` (its domain) pair with the contracted axes of `op2(a2)` (its
# codomain), and each pair must be a dual pair: `dual(ax1) == ax2`. `op` is `identity` or `conj`,
# and `conj` dualizes a graded axis. `dual` and `conj` fall back to the identity for plain ranges,
# so this reduces to `ax1 == ax2` for non-graded arrays.
function check_contracted_axes(op1, a1, perm1_domain, op2, a2, perm2_codomain)
    length(perm1_domain) == length(perm2_codomain) || throw(
        ArgumentError(
            "Number of contracted axes do not match: `a1` has $(length(perm1_domain)), `a2` has $(length(perm2_codomain))"
        )
    )
    for (i, j) in zip(perm1_domain, perm2_codomain)
        ax1 = op1(axes(a1, i))
        ax2 = op2(axes(a2, j))
        dual(ax1) == ax2 || throw(
            ArgumentError(
                "Contracted axes do not match: `$op1(axes(a1, $i)) = $ax1` and `$op2(axes(a2, $j)) = $ax2`"
            )
        )
    end
    return nothing
end

function check_input(
        ::typeof(contract),
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    check_biperm(a1, perm1_codomain, perm1_domain)
    check_biperm(a2, perm2_codomain, perm2_domain)
    check_contracted_axes(identity, a1, perm1_domain, identity, a2, perm2_codomain)
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

function check_input(
        ::typeof(contractpermop),
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    for op in (op1, op2)
        op === identity || op === conj ||
            throw(ArgumentError("`op` must be `identity` or `conj`, got `$op`"))
    end
    check_biperm(a1, perm1_codomain, perm1_domain)
    check_biperm(a2, perm2_codomain, perm2_domain)
    check_contracted_axes(op1, a1, perm1_domain, op2, a2, perm2_codomain)
    return nothing
end

function check_input(
        ::typeof(contractpermopadd!),
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    # TODO: FIXME: Check that uncontracted axes match.
    check_input(
        contractpermop,
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    check_biperm(a_dest, perm_dest_codomain, perm_dest_domain)
    return nothing
end

function output_axes(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    return output_axes(
        contractpermop,
        perm_dest_codomain, perm_dest_domain,
        identity, a1, perm1_codomain, perm1_domain,
        identity, a2, perm2_codomain, perm2_domain
    )
end
# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
function output_axes(
        ::typeof(contractpermop),
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    axes_codomain = map(i -> op1(axes(a1, i)), perm1_codomain)
    axes_domain = map(i -> op2(axes(a2, i)), perm2_domain)
    axes_uncontracted = (axes_codomain..., axes_domain...)
    axes_codomain_dest, axes_domain_dest = bipartition(
        axes_uncontracted, perm_dest_codomain, perm_dest_domain
    )
    # The operand axes are stored/dualized, so un-dualize the domain axes into the codomain-facing
    # construction convention shared by `similar_map` and `unmatricize` (a no-op on dense axes).
    return axes_codomain_dest, conj.(axes_domain_dest)
end

function allocate_output(
        ::typeof(contract),
        perm_dest_codomain, perm_dest_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    return allocate_output(
        contractpermop,
        perm_dest_codomain, perm_dest_domain,
        identity, a1, perm1_codomain, perm1_domain,
        identity, a2, perm2_codomain, perm2_domain
    )
end
# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
# The destination `contractpermop` writes into. A structured operand type overloads this directly,
# deriving the axes and element type from `output_axes` and `Base.promote_op` as below; the
# permutations are part of the signature because the contraction pattern is not recoverable from the
# destination leg counts alone.
function allocate_output(
        ::typeof(contractpermop),
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    check_input(
        contractpermop,
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    axes_codomain_dest, axes_domain_dest = output_axes(
        contractpermop,
        perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return zero!(similar_map(a1, T, axes_codomain_dest, axes_domain_dest))
end
