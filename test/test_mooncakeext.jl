using Mooncake: Mooncake
using Random: Random
using TensorAlgebra: AbstractContractAlgorithm, BiTuple, DefaultContractAlgorithm,
    MatricizeContract, allocate_output, biperm, biperms, check_input, contract, contract!,
    contract_labels, contractadd!, contractpermadd!, default_algorithm, select_algorithm
using Test: @test, @testset

@testset "MooncakeExt" begin
    elt = Float64
    mode = Mooncake.ReverseMode
    rng = Random.default_rng()
    is_primitive = false
    atol = eps(real(elt))^(3 / 4)
    rtol = eps(real(elt))^(3 / 4)
    @testset "zero derivatives" begin
        @test Mooncake.tangent_type(BiTuple) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(AbstractContractAlgorithm) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(DefaultContractAlgorithm) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(MatricizeContract) ≡ Mooncake.NoTangent

        dest = randn(elt, (2, 2))
        a1 = randn(elt, (2, 2))
        a2 = randn(elt, (2, 2))
        biperm_dest = BiTuple((1,), (2,))
        biperm1 = BiTuple((1,), (2,))
        biperm2 = BiTuple((1,), (2,))
        labels_dest = (:i, :k)
        labels1 = (:i, :j)
        labels2 = (:j, :k)

        Mooncake.TestUtils.test_rule(
            rng, allocate_output, contract, biperm_dest.t1, biperm_dest.t2, a1, biperm1.t1,
            biperm1.t2,
            a2, biperm2.t1, biperm2.t2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, biperm, (1, 2, 3), (1, 2), (3,); mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, biperms, contract, labels_dest, labels1, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, check_input, contract, a1, biperm1.t1, biperm1.t2, a2, biperm2.t1,
            biperm2.t2;
            mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, check_input, contract!, dest, biperm_dest.t1, biperm_dest.t2,
            a1, biperm1.t1, biperm1.t2, a2, biperm2.t1, biperm2.t2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, contract_labels, labels1, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, contract_labels, a1, labels1, a2, labels2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, default_algorithm, contract!, dest, a1, a2; mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, select_algorithm, contract!, DefaultContractAlgorithm(), dest, a1, a2;
            mode, is_primitive
        )
    end
    @testset "contract" begin
        α = true
        β = false
        @testset "contractpermadd! (BiTuple)" begin
            dest = randn(elt, (2, 2))
            a1 = randn(elt, (2, 2))
            a2 = randn(elt, (2, 2))
            biperm_dest = BiTuple((1,), (2,))
            biperm1 = BiTuple((1,), (2,))
            biperm2 = BiTuple((1,), (2,))
            Mooncake.TestUtils.test_rule(
                rng, contractpermadd!, dest, biperm_dest.t1, biperm_dest.t2,
                a1, biperm1.t1, biperm1.t2, a2, biperm2.t1, biperm2.t2, α, β;
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
