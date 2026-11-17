# Kept out of line, so a caller that takes two operands calls the copy compiled for each array type
# rather than carrying its own.
@noinline function check_biperm(a, perm_codomain, perm_domain)
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    isbiperm(perm_codomain, perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    return nothing
end

# The contracted axes of the two operands, in matching order, must pair up as duals:
# `dual(ax1) == ax2`, after conjugating each operand's axes under its flag (`conj` dualizes a graded
# axis). `dual` and `conj` fall back to the identity for plain ranges, so this reduces to
# `ax1 == ax2` for non-graded arrays. It takes only the contracted axes, not the arrays or
# permutations, and is kept out of line, so it compiles once per number and type of contracted axes
# rather than once per pair of operand types or contraction shape.
@noinline function check_contracted_axes(
        isconj1::Bool, axes1::Tuple, isconj2::Bool, axes2::Tuple
    )
    axes1 = isconj1 ? map(conj, axes1) : axes1
    axes2 = isconj2 ? map(conj, axes2) : axes2
    length(axes1) == length(axes2) || throw(
        ArgumentError(
            "Number of contracted axes do not match: `a1` has $(length(axes1)), `a2` has $(length(axes2))"
        )
    )
    for (k, (ax1, ax2)) in enumerate(zip(axes1, axes2))
        dual(ax1) == ax2 || throw(
            ArgumentError(
                "Contracted axes do not match: contracted pair $k is $ax1 in the first operand and $ax2 in the second"
            )
        )
    end
    return nothing
end

function check_input(
        ::typeof(contractpermconjadd!),
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain
    )
    # TODO: FIXME: Check that uncontracted axes match.
    check_biperm(a1, perm1_codomain, perm1_domain)
    check_biperm(a2, perm2_codomain, perm2_domain)
    check_contracted_axes(
        isconj1, map(i -> axes(a1, i), perm1_domain),
        isconj2, map(i -> axes(a2, i), perm2_codomain)
    )
    check_biperm(a_dest, perm_dest_codomain, perm_dest_domain)
    return nothing
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
# Takes the operands' axes rather than the arrays, and is kept out of line, so a structured
# operand's `allocate_output` reuses the copy compiled per axes type. The dense `allocate_output`
# below inlines it, since it is already compiled per contraction shape.
@noinline function output_axes(
        ::typeof(contractpermconj),
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, axes1::Tuple, perm1_codomain, perm1_domain,
        isconj2::Bool, axes2::Tuple, perm2_codomain, perm2_domain
    )
    # Conjugate each whole tuple once, outside the maps below, which compile once per shape.
    axes1 = isconj1 ? map(conj, axes1) : axes1
    axes2 = isconj2 ? map(conj, axes2) : axes2
    axes_codomain = map(i -> axes1[i], perm1_codomain)
    axes_domain = map(i -> axes2[i], perm2_domain)
    axes_uncontracted = (axes_codomain..., axes_domain...)
    axes_codomain_dest, axes_domain_dest = bipartition(
        axes_uncontracted, perm_dest_codomain, perm_dest_domain
    )
    # The operand axes are stored/dualized, so un-dualize the domain axes into the codomain-facing
    # construction convention shared by `similar_map` and `unmatricize` (a no-op on dense axes).
    return axes_codomain_dest, conj.(axes_domain_dest)
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
# The destination `contractpermconj` writes into. Inputs are validated by the contraction kernel that
# fills it, not here. A structured operand type overloads this directly,
# deriving the axes and element type from `output_axes` and `Base.promote_op` as below; the
# permutations are part of the signature because the contraction pattern is not recoverable from the
# destination leg counts alone.
function allocate_output(
        ::typeof(contractpermconj),
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain
    )
    axes_codomain_dest, axes_domain_dest = @inline output_axes(
        contractpermconj,
        perm_dest_codomain, perm_dest_domain,
        isconj1, axes(a1), perm1_codomain, perm1_domain,
        isconj2, axes(a2), perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return zero!(similar_map(a1, T, axes_codomain_dest, axes_domain_dest))
end
