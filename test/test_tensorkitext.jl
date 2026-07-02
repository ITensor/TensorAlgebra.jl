using StableRNGs: StableRNG
using TensorAlgebra: contract, matricize, similar_map, unmatricize
using TensorKit: @tensor, Rep, SU₂, U₁, fuse, isomorphism, randn, space, ←, ⊗
using Test: @test, @testset

# A shared bond contracts when it sits in one operand's domain and the other's codomain, i.e.
# `space(a, ka) == dual(space(b, kb))`, exactly as it would in a TensorKit tensor network.
@testset "TensorKitExt (eltype = $elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(1234)

    @testset "contract abelian, rank-2 bond" begin
        W = Rep[U₁](0 => 2, 1 => 1)
        X = Rep[U₁](0 => 1, 1 => 2)
        Y = Rep[U₁](-1 => 1, 0 => 2)
        a = randn(rng, elt, W, X)
        b = randn(rng, elt, X, Y)
        c, labels = contract(a, (:i, :j), b, (:j, :k))
        @test labels == [:i, :k]
        @test c ≈ a * b
    end

    @testset "contract abelian, rank-3 with permuted output" begin
        A1 = Rep[U₁](0 => 2, 1 => 1)
        A2 = Rep[U₁](0 => 1, 1 => 1)
        B = Rep[U₁](0 => 1, -1 => 2)
        C1 = Rep[U₁](0 => 2)
        C2 = Rep[U₁](1 => 1, 0 => 1)
        a = randn(rng, elt, A1 ⊗ A2, B)
        b = randn(rng, elt, B, C1 ⊗ C2)
        c, labels = contract(a, (:i, :j, :m), b, (:m, :k, :l))
        @test labels == [:i, :j, :k, :l]
        @tensor ref[i, j; k, l] := a[i, j, m] * b[m, k, l]
        @test c ≈ ref
    end

    @testset "contract non-abelian (SU2)" begin
        P = Rep[SU₂](1 // 2 => 1)
        Q = Rep[SU₂](0 => 1, 1 => 1)
        R = Rep[SU₂](1 // 2 => 2)
        s = randn(rng, elt, P ⊗ Q, R)
        w = randn(rng, elt, R, P)
        c, labels = contract(s, (:i, :j, :m), w, (:m, :k))
        @test labels == [:i, :j, :k]
        @tensor ref[i, j; k] := s[i, j, m] * w[m, k]
        @test c ≈ ref
    end

    @testset "matricize / unmatricize round-trip" begin
        A1 = Rep[U₁](0 => 2, 1 => 1)
        A2 = Rep[U₁](0 => 1, 1 => 1)
        B = Rep[U₁](0 => 1, -1 => 2)
        C1 = Rep[U₁](0 => 2)
        t = randn(rng, elt, A1 ⊗ A2, B ⊗ C1)
        codomain_axes = (space(t, 1), space(t, 2))
        # `unmatricize` takes the domain axes codomain-facing (un-dualized), so pass `B`, `C1`
        # directly rather than the dualized `space(t, 3)`, `space(t, 4)`.
        domain_axes = (B, C1)
        m = matricize(t, Val(2))
        @test space(m) == space(t)
        back = unmatricize(m, codomain_axes, domain_axes)
        @test back ≈ t
    end

    @testset "unmatricize splits combiner-style" begin
        for (V1, V2, U) in (
                (Rep[U₁](0 => 1, 1 => 2), Rep[U₁](0 => 2, -1 => 1), Rep[U₁](0 => 1, 1 => 1)),
                (
                    Rep[SU₂](1 // 2 => 1, 0 => 1),
                    Rep[SU₂](1 // 2 => 2),
                    Rep[SU₂](0 => 1, 1 => 1),
                ),
            )
            a = randn(rng, elt, V1 ⊗ V2, U)
            # Fuse the codomain into a single space, so the matrix codomain fuses to `(V1, V2)`
            # rather than matching it; `unmatricize` must split it back by rewrapping the data.
            m = isomorphism(elt, fuse(V1, V2), V1 ⊗ V2) * a
            @test space(m) != space(a)
            back = unmatricize(m, (space(a, 1), space(a, 2)), (U,))
            @test space(back) == space(a)
            @test back ≈ a
        end
    end

    @testset "similar_map space convention" begin
        A1 = Rep[U₁](0 => 2, 1 => 1)
        A2 = Rep[U₁](0 => 1, 1 => 1)
        B = Rep[U₁](0 => 1, -1 => 2)
        C1 = Rep[U₁](0 => 2)
        t = randn(rng, elt, A1 ⊗ A2, B ⊗ C1)
        sm = similar_map(t, elt, (A1, A2), (B, C1))
        @test space(sm) == space(t)

        # An all-codomain `TensorMap` (empty domain) is how ITensorBase direct-wraps a
        # `TensorMap`, so `similar_map` must handle empty axis tuples on either side.
        t_codomain = randn(rng, elt, (A1 ⊗ A2) ← one(A1))
        sm_codomain = similar_map(t_codomain, elt, (A1, A2), ())
        @test space(sm_codomain) == space(t_codomain)

        t_domain = randn(rng, elt, one(A1) ← (A1 ⊗ A2))
        sm_domain = similar_map(t_domain, elt, (), (A1, A2))
        @test space(sm_domain) == space(t_domain)
    end
end
