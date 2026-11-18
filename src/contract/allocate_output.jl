using LinearAlgebra: Adjoint, Transpose

# Kept out of line so it compiles per array type, not per pair of operands.
@noinline function check_biperm(a, perm_codomain, perm_domain)
    ndims(a) == length(perm_codomain) + length(perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    isbiperm(perm_codomain, perm_domain) ||
        throw(ArgumentError("Invalid bipartitioned permutation"))
    return nothing
end

# Kept out of line and given only the axes, so it compiles per axis type rather than per pair of
# operands.
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
"""
    output_axes(contractpermconjalign, args...) -> (axes_codomain, axes_domain)

The codomain and domain axes of the result of [`contractpermconjalign`](@ref) called with `args`.
"""
function output_axes end

# Given only the axes and kept out of line, so it compiles per axis type rather than per pair of
# operands.
@noinline function output_axes(
        ::typeof(contractpermconjalign),
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, axes1::Tuple, perm1_codomain, perm1_domain,
        isconj2::Bool, axes2::Tuple, perm2_codomain, perm2_domain
    )
    axes1 = isconj1 ? map(conj, axes1) : axes1
    axes2 = isconj2 ? map(conj, axes2) : axes2
    axes_codomain = map(i -> axes1[i], perm1_codomain)
    axes_domain = map(i -> axes2[i], perm2_domain)
    axes_uncontracted = (axes_codomain..., axes_domain...)
    axes_codomain_dest, axes_domain_dest = bipartition(
        axes_uncontracted, perm_dest_codomain, perm_dest_domain
    )
    # `similar_map` takes the domain axes un-dualized.
    return axes_codomain_dest, conj.(axes_domain_dest)
end

# TODO: Use `ArrayLayouts`-like `MulAdd` object,
# i.e. `ContractAdd`?
# Inputs are validated by the contraction itself.
function allocate_output(
        ::typeof(contractpermconjalign),
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain;
        allocator = DefaultAllocator()
    )
    axes_codomain_dest, axes_domain_dest = @inline output_axes(
        contractpermconjalign,
        perm_dest_codomain, perm_dest_domain,
        isconj1, axes(a1), perm1_codomain, perm1_domain,
        isconj2, axes(a2), perm2_codomain, perm2_domain
    )
    T = Base.promote_op(matprod, eltype(a1), eltype(a2))
    return zero!(similar_map(allocator, a1, T, axes_codomain_dest, axes_domain_dest))
end

# As `LinearAlgebra`'s generic `*` allocates its result. A transposed matrix allocates like its
# parent, which is passed on so that an allocator for the parent's type can serve it.
function allocate_output(
        ::typeof(*), m1::AbstractMatrix, m2::AbstractMatrix; allocator = DefaultAllocator()
    )
    T = Base.promote_op(matprod, eltype(m1), eltype(m2))
    prototype = m2 isa Union{Adjoint, Transpose} ? parent(m2) : m2
    return similar_map(allocator, prototype, T, (axes(m1, 1),), (axes(m2, 2),))
end
