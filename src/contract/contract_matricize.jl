using LinearAlgebra: mul!

# The matricized kernel for arrays whose matricization is a plain fold: matricize both operands,
# multiply, and write the product into the destination's matricization. An array family whose
# contraction needs more than the fold (a fermionic twist, a block-sparse product) owns its own
# `ContractAlgorithm` and `contractpermopadd!` method rather than hooking into this one.
function contractpermopadd!(
        ::MatricizeContract,
        a_dest::AbstractArray, biperm_dest_codomain, biperm_dest_domain,
        op1, a1::AbstractArray, biperm1_codomain, biperm1_domain,
        op2, a2::AbstractArray, biperm2_codomain, biperm2_domain,
        α::Number, β::Number
    )
    biperm_dest = (biperm_dest_codomain..., biperm_dest_domain...)
    invperm_codomain, invperm_domain =
        bipartition(invperm(biperm_dest), Val(length(biperm1_codomain)))
    check_input(
        contract!,
        a_dest, invperm_codomain, invperm_domain,
        a1, biperm1_codomain, biperm1_domain,
        a2, biperm2_codomain, biperm2_domain
    )
    a1_mat = matricizeop(op1, a1, biperm1_codomain, biperm1_domain)
    a2_mat = matricizeop(op2, a2, biperm2_codomain, biperm2_domain)
    if is_output_view(matricizeop, identity, a_dest, invperm_codomain, invperm_domain)
        # The matricization shares `a_dest`'s memory, so the matmul is the whole operation.
        a_dest_mat = matricizeopview(identity, a_dest, invperm_codomain, invperm_domain)
        mul!(a_dest_mat, a1_mat, a2_mat, α, β)
    else
        # Let the matmul allocate its matrix result and scatter it into `a_dest` with `α` and `β`
        # folded into the one permuted pass, so `a_dest` is never gathered. Every coupled-sector
        # block is materialized (the matmul zeros the ones it does not reach), so the scatter
        # reaches `a_dest` in full.
        a_dest_mat = a1_mat * a2_mat
        unmatricizeadd!(a_dest, a_dest_mat, invperm_codomain, invperm_domain, α, β)
    end
    return a_dest
end
