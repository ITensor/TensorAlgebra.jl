using LinearAlgebra: mul!
using TupleTools: TupleTools

# Matricize both operands, multiply, and write the product into the destination. There are four
# layouts: the contracted legs in `a1`'s or in `a2`'s order, times `a1 * a2` or the transposed
# `a2 * a1`. The one that copies the fewest elements is used.
function contractpermconjadd!(
        ::MatricizeContract,
        a_dest::AbstractArray, biperm_dest_codomain, biperm_dest_domain,
        isconj1::Bool, a1::AbstractArray, biperm1_codomain, biperm1_domain,
        isconj2::Bool, a2::AbstractArray, biperm2_codomain, biperm2_domain,
        α::Number, β::Number
    )
    biperm_dest = (biperm_dest_codomain..., biperm_dest_domain...)
    TupleTools.isperm(biperm_dest) ||
        throw(ArgumentError("destination permutation is not a permutation"))
    invperm_codomain, invperm_domain =
        bipartition(TupleTools.invperm(biperm_dest), Val(length(biperm1_codomain)))
    check_input(
        contractpermconjadd!,
        a_dest, invperm_codomain, invperm_domain,
        isconj1, a1, biperm1_codomain, biperm1_domain,
        isconj2, a2, biperm2_codomain, biperm2_domain
    )
    # The contracted legs reordered to `a2`'s storage order.
    p = TupleTools.sortperm(biperm2_codomain)
    biperm1_domain_alt = map(i -> biperm1_domain[i], p)
    biperm2_codomain_alt = map(i -> biperm2_codomain[i], p)
    l1 = matricize_copy_lengths(
        isconj1, a1, biperm1_codomain, biperm1_domain, biperm1_domain_alt
    )
    l2 = matricize_copy_lengths(
        isconj2, a2, biperm2_domain, biperm2_codomain, biperm2_codomain_alt
    )
    l_dest, l_dest_swapped = dest_copy_lengths(a_dest, invperm_codomain, invperm_domain)
    costs = (
        l_dest + l1[1] + l2[3],
        l_dest + l1[2] + l2[4],
        l_dest_swapped + l2[1] + l1[3],
        l_dest_swapped + l2[2] + l1[4],
    )
    i = argmin(costs)
    biperm1_contracted, biperm2_contracted = if isodd(i)
        (biperm1_domain, biperm2_codomain)
    else
        (biperm1_domain_alt, biperm2_codomain_alt)
    end
    if i ≤ 2
        _contractpermconjadd_matricized!(
            a_dest, invperm_codomain, invperm_domain,
            isconj1, a1, biperm1_codomain, biperm1_contracted,
            isconj2, a2, biperm2_contracted, biperm2_domain,
            α, β
        )
    else
        _contractpermconjadd_matricized!(
            a_dest, invperm_domain, invperm_codomain,
            isconj2, a2, biperm2_domain, biperm2_contracted,
            isconj1, a1, biperm1_contracted, biperm1_codomain,
            α, β
        )
    end
    return a_dest
end

# The scoring helpers each see one array and are kept out of line, so they compile per array type
# rather than per pair of operands.

# The number of elements copied to matricize `a` as the left factor `(open | contracted)` and as
# the right factor `(contracted | open)`, for both orders of the contracted legs.
@noinline function matricize_copy_lengths(
        isconj::Bool, a, perm_open, perm_contracted, perm_contracted_alt
    )
    n = length(a)
    copy_length(perm_codomain, perm_domain) =
        is_output_view(matricizeconj, isconj, a, perm_codomain, perm_domain) ? 0 : n
    return (
        copy_length(perm_open, perm_contracted),
        copy_length(perm_open, perm_contracted_alt),
        copy_length(perm_contracted, perm_open),
        copy_length(perm_contracted_alt, perm_open),
    )
end

# The number of elements copied to write the destination through its direct and its swapped split.
# Only a plain (not transposed) view is written through, since `mul!` into a lazy `transpose` does
# not use BLAS.
@noinline function dest_copy_lengths(a_dest, perm_codomain, perm_domain)
    n = length(a_dest)
    function copy_length(perm_codomain, perm_domain)
        isidentitybiperm(perm_codomain, perm_domain) &&
            is_output_view(matricizeconj, false, a_dest, perm_codomain, perm_domain) &&
            return 0
        return n
    end
    return copy_length(perm_codomain, perm_domain), copy_length(perm_domain, perm_codomain)
end

# `a_dest = β * a_dest + α * m1 * m2` for the operands matricized as `m1` and `m2`, with the
# product's rows and columns placed in `a_dest` by `(perm_dest_codomain, perm_dest_domain)`.
function _contractpermconjadd_matricized!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    a1_mat = matricizeconj(isconj1, a1, perm1_codomain, perm1_domain)
    a2_mat = matricizeconj(isconj2, a2, perm2_codomain, perm2_domain)
    if isidentitybiperm(perm_dest_codomain, perm_dest_domain) &&
            is_output_view(matricizeconj, false, a_dest, perm_dest_codomain, perm_dest_domain)
        a_dest_mat = matricizeconjview(false, a_dest, perm_dest_codomain, perm_dest_domain)
        # The matricizations are inferred as a `Union` of `Matrix`, `Transpose` and `Adjoint`, which
        # would make the compiler analyze the matrix product on that `Union`. Dispatching at run
        # time reaches the method for the concrete types, which is usually already compiled.
        Base.inferencebarrier(mul!)(a_dest_mat, a1_mat, a2_mat, α, β)
    else
        Base.inferencebarrier(unmatricizemuladd!)(
            a_dest, a1_mat, a2_mat, perm_dest_codomain, perm_dest_domain, α, β
        )
    end
    return a_dest
end

# `a_dest = β * a_dest + α * unmatricize(m1 * m2)`.
function unmatricizemuladd!(a_dest, m1, m2, perm_codomain, perm_domain, α, β)
    unmatricizeadd!(a_dest, m1 * m2, perm_codomain, perm_domain, α, β)
    return nothing
end
