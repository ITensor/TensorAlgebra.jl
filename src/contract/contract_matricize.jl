using LinearAlgebra: mul!

function contractopadd!(
        algorithm::Matricize,
        a_dest::AbstractArray, biperm_dest_codomain, biperm_dest_domain,
        op1, a1::AbstractArray, biperm1_codomain, biperm1_domain,
        op2, a2::AbstractArray, biperm2_codomain, biperm2_domain,
        α::Number, β::Number
    )
    return contractopadd!_matricize(
        algorithm,
        a_dest, biperm_dest_codomain, biperm_dest_domain,
        op1, a1, biperm1_codomain, biperm1_domain,
        op2, a2, biperm2_codomain, biperm2_domain,
        α, β
    )
end

function contractopadd!_matricize(
        algorithm::Matricize,
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        op1, a1::AbstractArray, perm1_codomain, perm1_domain,
        op2, a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    perm_dest = (perm_dest_codomain..., perm_dest_domain...)
    invperm_codomain, invperm_domain =
        blocks(biperm(invperm(perm_dest), length(perm1_codomain)))
    check_input(
        contract!,
        a_dest, invperm_codomain, invperm_domain,
        a1, perm1_codomain, perm1_domain,
        a2, perm2_codomain, perm2_domain
    )
    a1_mat = matricizeop(algorithm.fusion_style, op1, a1, perm1_codomain, perm1_domain)
    a2_mat = matricizeop(algorithm.fusion_style, op2, a2, perm2_codomain, perm2_domain)
    a_dest_mat = a1_mat * a2_mat
    unmatricizeadd!(
        algorithm.fusion_style, a_dest, a_dest_mat, invperm_codomain, invperm_domain, α, β
    )
    return a_dest
end
