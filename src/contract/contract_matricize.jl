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
        blocks(biperm(invperm(biperm_dest), length(biperm1_codomain)))
    check_input(
        contract!,
        a_dest, invperm_codomain, invperm_domain,
        a1, biperm1_codomain, biperm1_domain,
        a2, biperm2_codomain, biperm2_domain
    )
    a1_mat = matricizeop(algorithm.fusion_style, op1, a1, biperm1_codomain, biperm1_domain)
    a2_mat = matricizeop(algorithm.fusion_style, op2, a2, biperm2_codomain, biperm2_domain)
    a_dest_mat = a1_mat * a2_mat
    unmatricizeadd!(
        algorithm.fusion_style, a_dest, a_dest_mat, invperm_codomain, invperm_domain, α, β
    )
    return a_dest
end
