using LinearAlgebra: mul!
using TupleTools: TupleTools

# ===============================  operand matricization  ==================================
# A contraction operand can also be matricized without a copy when its stored split is the swapped
# one: the stored matricization's transpose, or its adjoint when `op` is `conj`. This is only for
# contraction operands (`mul!` takes the lazy wrappers), not for `matricizeop` in general, whose
# other callers expect the stored layout.

"""
    TensorAlgebra.contract_matricizeop(op, a, perm_codomain, perm_domain)

Matricize `a` as a contraction operand: like [`matricizeop`](@ref), but it may also return a
`transpose` or `adjoint` of the memory-sharing matricization at the swapped split. Copy-free
exactly when `is_output_view(contract_matricizeop, op, a, perm_codomain, perm_domain)`.
"""
function contract_matricizeop(op, a, perm_codomain, perm_domain)
    if !is_output_view(matricizeop, op, a, perm_codomain, perm_domain) &&
            is_output_view(contract_matricizeop, op, a, perm_codomain, perm_domain)
        m = matricizeopview(identity, a, perm_domain, perm_codomain)
        return op === conj ? adjoint(m) : transpose(m)
    end
    return matricizeop(op, a, perm_codomain, perm_domain)
end

# An array type that cannot represent the swapped split as a `transpose` (a graded array, whose
# transpose would reverse its arrows) overloads this to allow only the `adjoint`, or neither.
function is_output_view(::typeof(contract_matricizeop), op, a, perm_codomain, perm_domain)
    return is_output_view(matricizeop, op, a, perm_codomain, perm_domain) ||
        is_output_view(matricizeop, identity, a, perm_domain, perm_codomain)
end

# ===============================  matricized contraction  ==================================
# The matricized kernel for arrays whose matricization is a plain fold: matricize both operands,
# multiply, and write the product into the destination's matricization. An array family whose
# contraction needs more than the fold (a fermionic twist, a block-sparse product) owns its own
# `ContractAlgorithm` and `contractpermopadd!` method rather than hooking into this one.
#
# The contraction can be laid out four ways: the contracted legs in `a1`'s or in `a2`'s storage
# order, times `a1 * a2` or `a2 * a1` (the transposed product, written into the destination's
# swapped split). Each layout is scored by the number of elements it copies to matricize the
# operands and the destination, and the cheapest is used.
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
    costs = (
        matricize_copy_length(
            a_dest, invperm_codomain, invperm_domain,
            op1, a1, biperm1_codomain, biperm1_domain,
            op2, a2, biperm2_codomain, biperm2_domain
        ),
        matricize_copy_length(
            a_dest, invperm_codomain, invperm_domain,
            op1, a1, biperm1_codomain, biperm1_domain_alt,
            op2, a2, biperm2_codomain_alt, biperm2_domain
        ),
        matricize_copy_length(
            a_dest, invperm_domain, invperm_codomain,
            op2, a2, biperm2_domain, biperm2_codomain,
            op1, a1, biperm1_domain, biperm1_codomain
        ),
        matricize_copy_length(
            a_dest, invperm_domain, invperm_codomain,
            op2, a2, biperm2_domain, biperm2_codomain_alt,
            op1, a1, biperm1_domain_alt, biperm1_codomain
        ),
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

function matricize_copy_length(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    n = 0
    is_output_view(contract_matricizeop, op1, a1, perm1_codomain, perm1_domain) ||
        (n += length(a1))
    is_output_view(contract_matricizeop, op2, a2, perm2_codomain, perm2_domain) ||
        (n += length(a2))
    is_output_view(matricizeop, identity, a_dest, perm_dest_codomain, perm_dest_domain) ||
        (n += length(a_dest))
    return n
end

# `a_dest = β * a_dest + α * matricize(op1(a1)) * matricize(op2(a2))`, with the product's rows and
# columns placed in `a_dest` by the destination bipermutation.
function matricized_mul_add!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    a1_mat = contract_matricizeop(op1, a1, perm1_codomain, perm1_domain)
    a2_mat = contract_matricizeop(op2, a2, perm2_codomain, perm2_domain)
    if is_output_view(matricizeop, identity, a_dest, perm_dest_codomain, perm_dest_domain)
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
