using LinearAlgebra: Adjoint
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, MatricizeContract, contractperm, contractperm!,
    contractpermadd!, contractpermalign, contractpermopadd!
using Test: @test, @test_throws, @testset

# The bipermutation rungs of the contract ladder, called directly. The labels entry points only
# ever hand down the canonical `(1:k), (k+1:n)` destination split, so an unevenly split or
# permuted destination is reachable only through these signatures.
@testset "contract bipermutation ladder (eltype=$elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(1234)
    a1 = randn(rng, elt, 2, 3, 4)
    a2 = randn(rng, elt, 4, 5)
    p1 = ((1, 2), (3,))
    p2 = ((1,), (2,))
    # The free legs in their natural order: `a1`'s codomain, then `a2`'s domain.
    ref = reshape(reshape(a1, 6, 4) * a2, 2, 3, 5)
    # Destination bipermutations, starting with the canonical split the labels forms produce and
    # then the ones they cannot: every free leg on one side, and permuted leg orders.
    dests = (
        ((1, 2), (3,)),
        ((1, 2, 3), ()),
        ((), (1, 2, 3)),
        ((3, 1), (2,)),
        ((2,), (3, 1)),
        ((1,), (2, 3)),
    )
    destsize(perm) = map(i -> size(ref, i), perm)

    @testset "contractperm takes the canonical destination split" begin
        @test contractperm(a1, p1..., a2, p2...) ≈ ref
    end

    @testset "contractpermalign honors an arbitrary destination bipermutation" begin
        for (dest_codomain, dest_domain) in dests
            perm = (dest_codomain..., dest_domain...)
            got = contractpermalign(dest_codomain, dest_domain, a1, p1..., a2, p2...)
            @test size(got) == destsize(perm)
            @test got ≈ permutedims(ref, perm)
        end
        @test a1 == randn(StableRNG(1234), elt, 2, 3, 4) # operands untouched
    end

    @testset "contractperm! writes into the destination" begin
        for (dest_codomain, dest_domain) in dests
            perm = (dest_codomain..., dest_domain...)
            dest = fill(elt(7), destsize(perm))
            @test contractperm!(
                dest, dest_codomain, dest_domain, a1, p1..., a2, p2...
            ) === dest
            @test dest ≈ permutedims(ref, perm)
        end
    end

    @testset "contractpermadd! scales the contraction and the destination" begin
        dest_codomain, dest_domain = (3, 1), (2,)
        perm = (dest_codomain..., dest_domain...)
        dest = randn(rng, elt, destsize(perm))
        dest_before = copy(dest)
        α, β = elt(2), elt(3)
        @test contractpermadd!(
            dest, dest_codomain, dest_domain, a1, p1..., a2, p2..., α, β
        ) === dest
        @test dest ≈ α * permutedims(ref, perm) + β * dest_before
    end

    @testset "contractpermopadd! applies each operand's op" begin
        conj_ref = reshape(reshape(conj(a1), 6, 4) * conj(a2), 2, 3, 5)
        dest = fill(elt(7), 2, 3, 5)
        @test contractpermopadd!(
            dest, (1, 2), (3,), conj, a1, p1..., conj, a2, p2..., true, false
        ) === dest
        @test dest ≈ conj_ref
        # Only the first operand conjugated, on a permuted destination.
        half_ref = permutedims(reshape(reshape(conj(a1), 6, 4) * a2, 2, 3, 5), (3, 1, 2))
        dest_half = fill(elt(7), destsize((3, 1, 2)))
        contractpermopadd!(
            dest_half, (3, 1), (2,), conj, a1, p1..., identity, a2, p2..., true, false
        )
        @test dest_half ≈ half_ref
    end

    @testset "malformed destinations and algorithms are rejected" begin
        @test_throws ArgumentError contractpermalign(
            (1, 1), (3,), a1, p1..., a2, p2...
        )
        @test_throws DimensionMismatch contractperm!(
            zeros(elt, 2, 3, 6), (1, 2), (3,), a1, p1..., a2, p2...
        )
        @test_throws ArgumentError contractpermalign(
            (1, 2), (3,), a1, p1..., randn(rng, elt, 3, 5), p2...
        )
        @test_throws ArgumentError contractpermalign(
            (1, 2), (3,), a1, p1..., a2, p2...; alg = :not_an_algorithm
        )
    end

    @testset "a named algorithm reaches the same result" begin
        dest_codomain, dest_domain = (3, 1), (2,)
        expected = permutedims(ref, (dest_codomain..., dest_domain...))
        for alg in (TensorAlgebra.DefaultContractAlgorithm(), MatricizeContract())
            @test contractpermalign(
                dest_codomain, dest_domain, a1, p1..., a2, p2...; alg
            ) ≈ expected
        end
    end
end

# The matricized kernel lays the contraction out with the contracted legs in either operand's
# storage order and as `a1 * a2` or `a2 * a1`, so every storage order of the operands and the
# destination, with and without a conjugate, has to give the same result.
@testset "matricized contraction in every storage layout (eltype=$elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    rng = StableRNG(1234)
    # `a[i, j, k] b[k, j, l] -> c[i, l]`, with each array stored in a permuted order.
    a = randn(rng, elt, 2, 3, 4)
    b = randn(rng, elt, 4, 3, 5)
    for pa in ((1, 2, 3), (3, 2, 1), (2, 3, 1)), pb in ((1, 2, 3), (3, 2, 1), (2, 1, 3)),
            pc in ((1, 2), (2, 1)),
            (op1, op2) in ((identity, identity), (conj, identity), (conj, conj))

        ap, bp = permutedims(a, pa), permutedims(b, pb)
        # Positions of `a`'s and `b`'s logical legs in the stored arrays.
        ia, ib = invperm(pa), invperm(pb)
        dest = zeros(elt, (2, 5)[collect(pc)]...)
        contractpermopadd!(
            dest, (pc[1],), (pc[2],),
            op1, ap, (ia[1],), (ia[2], ia[3]),
            op2, bp, (ib[2], ib[1]), (ib[3],),
            true, false
        )
        expected = reshape(op1.(a), 2, 12) * reshape(permutedims(op2.(b), (2, 1, 3)), 12, 5)
        @test dest ≈ permutedims(expected, pc)
    end
end

@testset "contract_matricizeop takes the swapped split lazily" begin
    a = randn(ComplexF64, 3, 4, 5)
    pc, pd = (3,), (1, 2)
    @test TensorAlgebra.contract_matricizeop(identity, a, pc, pd) ==
        transpose(reshape(a, 12, 5))
    m = TensorAlgebra.contract_matricizeop(conj, a, pc, pd)
    @test m isa Adjoint
    @test m == reshape(a, 12, 5)'
    @test TensorAlgebra.contract_matricizeop(conj, a, (1, 2), (3,)) ==
        conj(reshape(a, 12, 5))
    @test !TensorAlgebra.is_output_view(
        TensorAlgebra.contract_matricizeop,
        identity,
        a,
        (2,),
        (1, 3)
    )
end
