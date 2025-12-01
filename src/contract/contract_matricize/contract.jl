using LinearAlgebra: mul!

function contractadd!(
        alg::Matricize,
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    return contractadd!_matricize(alg, a_dest, biperm_dest, a1, biperm1, a2, biperm2, α, β)
end

function contractadd!_matricize(
        alg::Matricize,
        a_dest::AbstractArray, biperm_dest::AbstractBlockPermutation{2},
        a1::AbstractArray, biperm1::AbstractBlockPermutation{2},
        a2::AbstractArray, biperm2::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    invbiperm = biperm(invperm(biperm_dest), length_codomain(biperm1))
    check_input(contract, a_dest, invbiperm, a1, biperm1, a2, biperm2)
    a1_mat = matricize(alg.fusion_style, a1, biperm1)
    a2_mat = matricize(alg.fusion_style, a2, biperm2)
    a_dest_mat = a1_mat * a2_mat
    unmatricizeadd!(alg.fusion_style, a_dest, a_dest_mat, invbiperm, α, β)
    return a_dest
end
