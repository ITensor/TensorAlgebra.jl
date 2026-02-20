using BlockArrays: blocks
using Mooncake: Mooncake
using Random: Random
using TensorAlgebra: AbstractBlockPermutation, BlockedPermutation, ContractAlgorithm,
    DefaultContractAlgorithm, Matricize, allocate_output, biperm, blockedperms, check_input,
    contract, contract!, contract_labels, contractadd!, default_contract_algorithm,
    permmortar, select_contract_algorithm
using Test: @test, @testset

@testset "MooncakeExt" begin
    elt = Float64
    mode = Mooncake.ReverseMode
    rng = Random.default_rng()
    is_primitive = false
    atol = eps(real(elt))^(3 / 4)
    rtol = eps(real(elt))^(3 / 4)
    @testset "zero derivatives" begin
        @test Mooncake.tangent_type(AbstractBlockPermutation) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(BlockedPermutation) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(ContractAlgorithm) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(DefaultContractAlgorithm) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(Matricize) ≡ Mooncake.NoTangent

        dest = randn(elt, (2, 2))
        a1 = randn(elt, (2, 2))
        a2 = randn(elt, (2, 2))
        biperm_dest = permmortar(((1,), (2,)))
        biperm1 = permmortar(((1,), (2,)))
        biperm2 = permmortar(((1,), (2,)))
        labels_dest = (:i, :k)
        labels1 = (:i, :j)
        labels2 = (:j, :k)

        Mooncake.TestUtils.test_rule(
            rng, allocate_output, contract, blocks(biperm_dest)..., a1, blocks(biperm1)...,
            a2, blocks(biperm2)...; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(rng, biperm, (1, 2, 3), Val(2); mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, biperm, (1, 2, 3), 2; mode, is_primitive)
        Mooncake.TestUtils.test_rule(
            rng, blockedperms, contract, labels_dest, labels1, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, check_input, contract, a1, blocks(biperm1)..., a2, blocks(biperm2)...;
            mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, check_input, contract!, dest, blocks(biperm_dest)...,
            a1, blocks(biperm1)..., a2, blocks(biperm2)...; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, contract_labels, labels1, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, contract_labels, a1, labels1, a2, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, default_contract_algorithm, a1, a2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, select_contract_algorithm, DefaultContractAlgorithm(), a1, a2;
            mode, is_primitive
        )
    end
    @testset "contract" begin
        α = true
        β = false
        @testset "contractadd! (BlockedPermutation)" begin
            dest = randn(elt, (2, 2))
            a1 = randn(elt, (2, 2))
            a2 = randn(elt, (2, 2))
            biperm_dest = permmortar(((1,), (2,)))
            biperm1 = permmortar(((1,), (2,)))
            biperm2 = permmortar(((1,), (2,)))
            Mooncake.TestUtils.test_rule(
                rng, contractadd!, dest, blocks(biperm_dest)...,
                a1, blocks(biperm1)..., a2, blocks(biperm2)..., α, β;
                atol, rtol, mode, is_primitive
            )
        end
        @testset "contractadd! (labels)" begin
            dest = randn(elt, (2, 2))
            a1 = randn(elt, (2, 2))
            a2 = randn(elt, (2, 2))
            labels_dest = (:i, :k)
            labels1 = (:i, :j)
            labels2 = (:j, :k)
            Mooncake.TestUtils.test_rule(
                rng, contractadd!, dest, labels_dest, a1, labels1, a2, labels2, α, β;
                atol, rtol, mode, is_primitive
            )
        end
        @testset "contract! (labels)" begin
            dest = randn(elt, (2, 2))
            a1 = randn(elt, (2, 2))
            a2 = randn(elt, (2, 2))
            labels_dest = (:i, :k)
            labels1 = (:i, :j)
            labels2 = (:j, :k)
            Mooncake.TestUtils.test_rule(
                rng, contract!, dest, labels_dest, a1, labels1, a2, labels2;
                atol, rtol, mode, is_primitive
            )
        end
        @testset "contract (labels)" begin
            a1 = randn(elt, (2, 2))
            a2 = randn(elt, (2, 2))
            labels1 = (:i, :j)
            labels2 = (:j, :k)
            Mooncake.TestUtils.test_rule(
                rng, contract, a1, labels1, a2, labels2; atol, rtol, mode, is_primitive
            )
        end
    end
end
