using LinearAlgebra: mul!
using TupleTools: TupleTools

# ===============================  matricized contraction  ==================================
# The matricized kernel: matricize both operands, multiply, and write the product into the
# destination's matricization.
#
# The contraction can be laid out four ways: the contracted legs in `a1`'s or in `a2`'s storage
# order, times `a1 * a2` or `a2 * a1` (the transposed product, written into the destination's
# swapped split). Each layout is scored by the number of elements it copies to matricize the
# operands and the destination, and the cheapest is passed to `matricized_mul_add!`. An array
# family whose product needs more than the fold (a fermionic twist on the contracted legs)
# overloads `matricized_mul_add!` and keeps this layout choice. The score ignores any copy that
# extra work forces.
# Specialized on the operations for the same reason as `check_input(contractpermop, ...)`.
function contractpermopadd!(
        ::MatricizeContract,
        a_dest::AbstractArray, biperm_dest_codomain, biperm_dest_domain,
        op1::Op1, a1::AbstractArray, biperm1_codomain, biperm1_domain,
        op2::Op2, a2::AbstractArray, biperm2_codomain, biperm2_domain,
        α::Number, β::Number
    ) where {Op1, Op2}
    biperm_dest = (biperm_dest_codomain..., biperm_dest_domain...)
    invperm_codomain, invperm_domain =
        bipartition(invperm(biperm_dest), Val(length(biperm1_codomain)))
    check_input(
        contractpermopadd!,
        a_dest, invperm_codomain, invperm_domain,
        op1, a1, biperm1_codomain, biperm1_domain,
        op2, a2, biperm2_codomain, biperm2_domain
    )
    # The contracted legs reordered to `a2`'s storage order.
    p = TupleTools.sortperm(biperm2_codomain)
    biperm1_domain_alt = map(i -> biperm1_domain[i], p)
    biperm2_codomain_alt = map(i -> biperm2_codomain[i], p)
    l1 = matricize_copy_lengths(
        op1, a1, biperm1_codomain, biperm1_domain, biperm1_domain_alt
    )
    l2 = matricize_copy_lengths(
        op2, a2, biperm2_domain, biperm2_codomain, biperm2_codomain_alt
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
        matricized_mul_add!(
            a_dest, invperm_codomain, invperm_domain,
            op1, a1, biperm1_codomain, biperm1_contracted,
            op2, a2, biperm2_contracted, biperm2_domain,
            α, β
        )
    else
        matricized_mul_add!(
            a_dest, invperm_domain, invperm_codomain,
            op2, a2, biperm2_domain, biperm2_contracted,
            op1, a1, biperm1_contracted, biperm1_codomain,
            α, β
        )
    end
    return a_dest
end

# The scoring helpers below each see one array, and are not inlined, so they compile once per
# array type rather than once per pair of operands.

# The number of elements copied to matricize `op(a)` as the left factor `(open | contracted)` and
# as the right factor `(contracted | open)`, for both orders of the contracted legs: none when the
# matricization is a view.
@noinline function matricize_copy_lengths(
        op, a, perm_open, perm_contracted, perm_contracted_alt
    )
    n = length(a)
    copy_length(perm_codomain, perm_domain) =
        is_output_view(matricizeop, op, a, perm_codomain, perm_domain) ? 0 : n
    return (
        copy_length(perm_open, perm_contracted),
        copy_length(perm_open, perm_contracted_alt),
        copy_length(perm_contracted, perm_open),
        copy_length(perm_contracted_alt, perm_open),
    )
end

# The number of elements copied to write the destination through its direct and its swapped
# split. The destination is written through its view only when that view is the direct one,
# since `mul!` into a lazy `transpose` falls off BLAS. A lazy view only arises at a non-identity
# bipermutation, and the transposed-product layout covers that one with a direct destination.
@noinline function dest_copy_lengths(a_dest, perm_codomain, perm_domain)
    n = length(a_dest)
    function copy_length(perm_codomain, perm_domain)
        isidentitybiperm(perm_codomain, perm_domain) &&
            is_output_view(matricizeop, identity, a_dest, perm_codomain, perm_domain) &&
            return 0
        return n
    end
    return copy_length(perm_codomain, perm_domain), copy_length(perm_domain, perm_codomain)
end

"""
    matricized_mul_add!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α, β
    )

Set `a_dest = β * a_dest + α * m1 * m2`, where `m1` and `m2` are the operands matricized with
`matricizeop(op1, a1, perm1_codomain, perm1_domain)` and
`matricizeop(op2, a2, perm2_codomain, perm2_domain)`, and the product's rows and columns are
placed in `a_dest` by the destination bipermutation `(perm_dest_codomain, perm_dest_domain)`.

The body of the `MatricizeContract` kernel for one storage layout. The kernel calls it with the
operands in either order, so an overload treats `a1` and `a2` as the left and right factor rather
than as the contraction's first and second operand.
"""
function matricized_mul_add!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    a1_mat = matricizeop(op1, a1, perm1_codomain, perm1_domain)
    a2_mat = matricizeop(op2, a2, perm2_codomain, perm2_domain)
    if isidentitybiperm(perm_dest_codomain, perm_dest_domain) &&
            is_output_view(matricizeop, identity, a_dest, perm_dest_codomain, perm_dest_domain)
        # The matricization shares `a_dest`'s memory, so the matmul is the whole operation.
        a_dest_mat = matricizeopview(identity, a_dest, perm_dest_codomain, perm_dest_domain)
        mul!(a_dest_mat, a1_mat, a2_mat, α, β)
    else
        # Let the matmul allocate its matrix result and scatter it into `a_dest` with `α` and `β`
        # folded into the one permuted pass, so `a_dest` is never gathered. Every coupled-sector
        # block is materialized (the matmul zeros the ones it does not reach), so the scatter
        # reaches `a_dest` in full.
        a_dest_mat = a1_mat * a2_mat
        unmatricizeadd!(a_dest, a_dest_mat, perm_dest_codomain, perm_dest_domain, α, β)
    end
    return a_dest
end
