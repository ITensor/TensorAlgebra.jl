using Base.Broadcast: broadcasted
using LinearAlgebra: norm
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, checked_project, contract, matricize, project,
    project_map, projectto!, rand_map, randn_map, similar_map, tryflattenlinear,
    unmatricize, zeros_map
using TensorKit: @tensor, AbstractTensorMap, Rep, SU₂, TensorMap, U₁, dual, fuse,
    isomorphism, randn, space, storagetype, ←, ⊗
using Test: @test, @test_throws, @testset

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

    @testset "unmatricize rejects a mismatched split" begin
        for (V1, V2, U) in (
                (Rep[U₁](0 => 1, 1 => 2), Rep[U₁](0 => 2, -1 => 1), Rep[U₁](0 => 1, 1 => 1)),
                (
                    Rep[SU₂](1 // 2 => 1, 0 => 1),
                    Rep[SU₂](1 // 2 => 2),
                    Rep[SU₂](0 => 1, 1 => 1),
                ),
            )
            a = randn(rng, elt, V1 ⊗ V2, U)
            # Fuse the codomain into a single space so the matrix codomain no longer splits into
            # `(V1, V2)`; `unmatricize` is a strict no-op, so it rejects the fused split rather
            # than rewrapping the data.
            m = isomorphism(elt, fuse(V1, V2), V1 ⊗ V2) * a
            @test space(m) != space(a)
            @test_throws ArgumentError unmatricize(m, (V1, V2), (U,))
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

    # The map constructors build a `TensorMap` from the codomain/domain spaces directly rather
    # than flattening, mirroring the `similar_map` space convention (domain given un-dualized).
    @testset "map construction on spaces" begin
        A1 = Rep[U₁](0 => 2, 1 => 1)
        A2 = Rep[U₁](0 => 1, 1 => 1)
        B = Rep[U₁](0 => 1, -1 => 2)

        for f in (randn_map, rand_map)
            t = f(rng, elt, (A1, A2), (B,))
            @test t isa AbstractTensorMap
            @test space(t) == ((A1 ⊗ A2) ← B)
        end
        z = zeros_map(elt, (A1, A2), (B,))
        @test z isa AbstractTensorMap
        @test space(z) == ((A1 ⊗ A2) ← B)
        @test iszero(z)

        # An empty domain gives an all-codomain `TensorMap`, the plain-tensor case.
        tc = randn_map(rng, elt, (A1, A2), ())
        @test space(tc) == ((A1 ⊗ A2) ← one(A1))

        # An empty codomain is the mirror case: the space type comes from the domain.
        td = randn_map(rng, elt, (), (A1, A2))
        @test space(td) == (one(A1) ← (A1 ⊗ A2))
        zd = zeros_map(elt, (), (A1, B))
        @test space(zd) == (one(A1) ← (A1 ⊗ B))
    end

    # `project` builds a `TensorMap` from a dense matrix: `similar_map` allocates a same-device
    # buffer (its block storage type follows the dense prototype) and `projectto!` fills the
    # symmetry-allowed blocks via `project_symmetric!`, discarding the rest. A charge-preserving
    # matrix survives; a charge-breaking one is projected away, and `checked_project` rejects that
    # loss.
    @testset "project a dense matrix into a TensorMap" begin
        W = Rep[U₁](0 => 1, 1 => 1)
        Sz = elt[0.5 0; 0 -0.5]
        Sx = elt[0 0.5; 0.5 0]

        pz = project(Sz, (W,), (W,))
        @test pz isa AbstractTensorMap
        @test space(pz) == (W ← W)
        @test pz ≈ TensorMap(Sz, W ← W)
        # `project` forwards to the `project_map` hook
        @test project_map(Sz, (W,), (W,)) ≈ pz
        # the block storage type follows the dense prototype's array type (device-preserving)
        @test storagetype(pz) == Vector{elt}

        # `projectto!` into a same-space buffer agrees with `project`
        @test projectto!(similar_map(Sz, elt, (W,), (W,)), Sz) ≈ pz

        # `checked_project` accepts the charge-preserving matrix (nothing discarded)
        @test checked_project(Sz, (W,), (W,)) ≈ pz
        # a charge-breaking matrix is projected to zero; `checked_project` rejects the discard
        @test norm(project(Sx, (W,), (W,))) == 0
        @test_throws InexactError checked_project(Sx, (W,), (W,); atol = 0, rtol = 0)

        # the flat two-argument form builds an all-codomain `TensorMap` (empty domain): only
        # the trivial-charge component of a dense vector survives the projection
        pv = project(elt[1, 0], (W,))
        @test pv isa AbstractTensorMap
        @test space(pv) == (W ← one(W))
        @test norm(pv) ≈ 1
        @test norm(project(elt[0, 1], (W,))) == 0
    end

    # `permutedims` reorders a `TensorMap`'s indices; the flat form gives an all-codomain
    # result, and the bipartition form re-expresses the requested codomain/domain split. Both
    # ride TensorKit's `permute` through the `bipermutedimsopadd!` interface, no dedicated method.
    @testset "permutedims on a TensorMap" begin
        A1 = Rep[U₁](0 => 2, 1 => 1)
        A2 = Rep[U₁](0 => 1, 1 => 1)
        B = Rep[U₁](0 => 1, -1 => 2)
        t = randn(rng, elt, A1 ⊗ A2, B)
        ref = permutedims(convert(Array, t), (3, 1, 2))

        # Flat: all-codomain result whose dense form matches the plain permutation.
        tf = TensorAlgebra.permutedims(t, (3, 1, 2))
        @test space(tf) == ((dual(B) ⊗ A1 ⊗ A2) ← one(A1))
        @test convert(Array, tf) == ref

        # Bipartition selecting the original split reproduces the space and data exactly.
        tb = TensorAlgebra.permutedims(t, (1, 2), (3,))
        @test space(tb) == space(t)
        @test convert(Array, tb) == convert(Array, t)

        # Repartitioning form: move the domain index into the codomain while reordering.
        tr = TensorAlgebra.permutedims(t, (3, 1), (2,))
        @test space(tr) == ((dual(B) ⊗ A1) ← dual(A2))
        @test convert(Array, tr) == ref

        # In-place form writes into a matching destination.
        dest = similar_map(t, elt, (dual(B), A1, A2), ())
        @test TensorAlgebra.permutedims!(dest, t, (3, 1, 2)) === dest
        @test convert(Array, dest) == ref
    end

    # A linear combination of `TensorMap`s flattens to a `LinearBroadcasted` that materializes
    # into a `TensorMap` destination via `copyto!`; a nonlinear broadcast has no linear form.
    @testset "linear-combination broadcast" begin
        V = Rep[SU₂](0 => 1, 1 // 2 => 2)
        a = randn(rng, elt, V ← V)
        b = randn(rng, elt, V ← V)

        lb = tryflattenlinear(broadcasted(+, a, broadcasted(*, 2, b)))
        dest = similar(a)
        copyto!(dest, lb)
        @test dest ≈ a + 2 * b

        # A nonlinear (element-wise) broadcast is not expressible as a `LinearBroadcasted`.
        @test isnothing(tryflattenlinear(broadcasted(*, a, b)))
        @test_throws ErrorException copy(broadcasted(*, a, b))
    end
end
