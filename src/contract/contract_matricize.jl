using LinearAlgebra: mul!

# The kernel's one seam where a backend prepares both operands together: GradedArrays twists
# the right factor of a fermionic contraction before matricizing it. Keyed on the rung whose
# arguments these are.
function matricize_inputs(
        ::typeof(contractpermopadd!),
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain
    )
    return matricizeop(op1, a1, perm1_codomain, perm1_domain),
        matricizeop(op2, a2, perm2_codomain, perm2_domain)
end

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
    a1_mat, a2_mat = matricize_inputs(
        contractpermopadd!,
        op1, a1, biperm1_codomain, biperm1_domain,
        op2, a2, biperm2_codomain, biperm2_domain
    )
    if is_output_view(matricizeop, identity, a_dest, invperm_codomain, invperm_domain)
        # The matricization shares `a_dest`'s memory, so the matmul is the whole operation.
        a_dest_mat = matricizeopview(identity, a_dest, invperm_codomain, invperm_domain)
        mul!(a_dest_mat, a1_mat, a2_mat, α, β)
    elseif iszero(β)
        # `β` is a strong zero, so `a_dest`'s current data is irrelevant: let the matmul
        # allocate its matrix result and scatter it into `a_dest`. Every coupled-sector block
        # is materialized (the matmul zeros the ones it does not reach), so the scatter
        # overwrites `a_dest` in full.
        a_dest_mat = a1_mat * a2_mat
        isone(α) || scale!(a_dest_mat, α)
        unmatricize!(a_dest, a_dest_mat, invperm_codomain, invperm_domain)
    else
        # `a_dest`'s data contributes through `β`, so gather it, multiply into the gathered
        # copy, and scatter back.
        a_dest_mat = matricizeopcopy(identity, a_dest, invperm_codomain, invperm_domain)
        mul!(a_dest_mat, a1_mat, a2_mat, α, β)
        unmatricize!(a_dest, a_dest_mat, invperm_codomain, invperm_domain)
    end
    return a_dest
end
