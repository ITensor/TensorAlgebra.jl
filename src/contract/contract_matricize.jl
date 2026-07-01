using LinearAlgebra: mul!

function contractopadd!(
        algorithm::Matricize,
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
    a1_mat = matricizeopperm(
        algorithm.left_fusion_style, op1, a1, biperm1_codomain, biperm1_domain
    )
    a2_mat = matricizeopperm(
        algorithm.right_fusion_style, op2, a2, biperm2_codomain, biperm2_domain
    )
    output_style = algorithm.output_fusion_style
    # Matricize the destination and multiply straight into it: a no-op for an aligned or
    # transposed dense output (a view aliasing `a_dest`, so `mul!` writes through and we
    # are done), a fresh permuted copy otherwise. Either way `matricize` seeds `a_dest_mat`
    # with `a_dest`'s current contents, so `β` rides on the `mul!` and a detached copy is
    # written back with a plain overwrite.
    a_dest_mat = matricizeperm(output_style, a_dest, invperm_codomain, invperm_domain)
    mul!(a_dest_mat, a1_mat, a2_mat, α, β)
    if !Base.mightalias(a_dest_mat, a_dest)
        unmatricizeperm!(output_style, a_dest, a_dest_mat, invperm_codomain, invperm_domain)
    end
    return a_dest
end
