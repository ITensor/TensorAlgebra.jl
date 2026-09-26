using Base.Broadcast: broadcasted
using LinearAlgebra: LinearAlgebra, norm
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, contract, contractalign, eig_full, eig_vals, eigh_full,
    eigh_vals, has_bipartition, left_null, left_orth, left_polar, lq_compact, lq_full,
    matricize, project, project_aux, projectto!, qr_compact, qr_full, rand_map, randn_map,
    right_null, right_orth, right_polar, similar_map, svd_compact, svd_full, svd_vals,
    tryflattenlinear, tryproject, unchecked_project, unmatricize, zeros_map
using TensorKit: TensorKit, @tensor, AbstractTensorMap, DiagonalTensorMap, Irrep, Rep, SU₂,
    TensorMap, U₁, dim, dual, fuse, isomorphism, randn, reduceddim, space, storagetype, ←, ⊗
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
        axes_codomain = (space(t, 1), space(t, 2))
        # `unmatricize` takes the domain axes codomain-facing (un-dualized), so pass `B`, `C1`
        # directly rather than the dualized `space(t, 3)`, `space(t, 4)`.
        axes_domain = (B, C1)
        m = matricize(t, (1, 2), (3, 4))
        @test space(m) == space(t)
        back = unmatricize(m, axes_codomain, axes_domain)
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

    # `project` builds a `TensorMap` from a dense matrix: `allocate_project` allocates a
    # same-device buffer (its block storage type follows the dense prototype) and `projectto!`
    # fills the symmetry-allowed blocks via `project_symmetric!`, discarding the rest. A
    # charge-preserving matrix survives; a charge-breaking one is projected away, which
    # `unchecked_project` allows silently and `project` rejects.
    @testset "project a dense matrix into a TensorMap" begin
        W = Rep[U₁](0 => 1, 1 => 1)
        Sz = elt[0.5 0; 0 -0.5]
        Sx = elt[0 0.5; 0.5 0]

        pz = project(Sz, (W,), (W,))
        @test pz isa AbstractTensorMap
        @test space(pz) == (W ← W)
        @test pz ≈ TensorMap(Sz, W ← W)
        # the block storage type follows the dense prototype's array type (device-preserving)
        @test storagetype(pz) == Vector{elt}

        # `projectto!` into a same-space buffer agrees with `project`
        @test projectto!(similar_map(Sz, elt, (W,), (W,)), Sz) ≈ pz

        # a charge-breaking matrix is projected to zero by `unchecked_project`; the checked
        # `project` rejects the discard
        @test norm(unchecked_project(Sx, (W,), (W,))) == 0
        @test_throws InexactError project(Sx, (W,), (W,); atol = 0, rtol = 0)

        # the flat two-argument form builds an all-codomain `TensorMap` (empty domain): only
        # the trivial-charge component of a dense vector survives the projection
        pv = project(elt[1, 0], (W,))
        @test pv isa AbstractTensorMap
        @test space(pv) == (W ← one(W))
        @test norm(pv) ≈ 1
        @test norm(unchecked_project(elt[0, 1], (W,))) == 0

        # a flat vector may omit a trailing length-1 auxiliary axis from its rank:
        # `project` appends it. An aux index carrying the canceling charge lets a
        # charged (equivariant) component survive the projection.
        aux = Rep[U₁](-1 => 1)
        pa = project(elt[0, 1], (W, aux))
        @test pa isa AbstractTensorMap
        @test space(pa) == ((W ⊗ aux) ← one(W))
        @test norm(pa) ≈ 1

        # only trailing *length-1* axes may be omitted: a longer trailing axis is a genuine
        # size mismatch that the block projection still rejects
        @test_throws DimensionMismatch project(elt[0, 1], (W, Rep[U₁](-1 => 3)))
    end

    # `project_aux` appends a derived auxiliary leg (as the last domain axis, matching the shape
    # of `raw`) whose space it derives, so the result is symmetry-allowed. `project` itself is
    # strict and rejects the surplus axis. The derivation scans the aux axis against the operator
    # content and works for non-abelian symmetries and multi-sector (direct-sum) auxes; the
    # abelian single-charge case falls out.
    @testset "project_aux derives the auxiliary space" begin
        # SU(2): a spin operator (aux = spin-1, dim 3) is recovered from its dense components.
        Ws = Rep[SU₂](1 // 2 => 1)
        ts = randn(rng, elt, Ws, Ws ⊗ Rep[SU₂](1 => 1))
        rs = project_aux(convert(Array, ts), (Ws,), (Ws,))
        @test space(rs) == (Ws ← (Ws ⊗ Rep[SU₂](1 => 1)))
        @test rs ≈ ts

        # strict `project` rejects that same surplus axis
        @test_throws ArgumentError project(convert(Array, ts), (Ws,), (Ws,))

        # U(1): a charge-shifting operator (non-self-dual aux = charge +1) is recovered.
        Wu = Rep[U₁](0 => 1, 1 => 1)
        tu = randn(rng, elt, Wu, Wu ⊗ Rep[U₁](1 => 1))
        ru = project_aux(convert(Array, tu), (Wu,), (Wu,))
        @test space(ru) == (Wu ← (Wu ⊗ Rep[U₁](1 => 1)))
        @test ru ≈ tu

        # U(1) direct sum (an MPO-style virtual leg): each slice carries its own charge.
        tds = randn(rng, elt, Wu, Wu ⊗ Rep[U₁](1 => 1, -1 => 1))
        rds = project_aux(convert(Array, tds), (Wu,), (Wu,))
        @test space(rds) == (Wu ← (Wu ⊗ Rep[U₁](1 => 1, -1 => 1)))
        @test rds ≈ tds

        # SU(2) direct sum of different irreps (scalar ⊕ vector part, dim 4).
        tmx = randn(rng, elt, Ws, Ws ⊗ Rep[SU₂](0 => 1, 1 => 1))
        rmx = project_aux(convert(Array, tmx), (Ws,), (Ws,))
        @test space(rmx) == (Ws ← (Ws ⊗ Rep[SU₂](0 => 1, 1 => 1)))
        @test rmx ≈ tmx

        # SU(2) multiplicity > 1: two spin-1 copies (dim 6).
        tm2 = randn(rng, elt, Ws, Ws ⊗ Rep[SU₂](1 => 2))
        rm2 = project_aux(convert(Array, tm2), (Ws,), (Ws,))
        @test space(rm2) == (Ws ← (Ws ⊗ Rep[SU₂](1 => 2)))
        @test rm2 ≈ tm2

        # data not covariant with any aux decomposition of the surplus axis is rejected
        @test_throws ArgumentError project_aux(randn(rng, elt, 2, 2, 3), (Ws,), (Ws,))

        # only one trailing surplus axis is supported: more is an error, not a flattening
        @test_throws ArgumentError project_aux(
            reshape(convert(Array, ts), 2, 2, 3, 1), (Ws,), (Ws,)
        )

        # a lower-rank `raw` that omits explicitly-given trailing length-1 axes is the
        # trailing-axes tolerance handled by plain `project`, not a surplus axis: it pads, it
        # does not derive. A domain aux of charge +1 admits the charge-1 component.
        aux1 = Rep[U₁](1 => 1)
        po = project(elt[0, 1], (Wu,), (aux1,))
        @test space(po) == (Wu ← aux1)
        @test norm(po) ≈ 1

        # the flat all-codomain (state) form also derives: a stack of basis states with
        # different charges gets a multi-sector aux
        ps = project_aux(elt[1 0; 0 1], (Wu,))
        @test space(ps) == (Wu ← Rep[U₁](0 => 1, 1 => 1))

        # `tryproject` gives `nothing` instead of throwing when the data is not invariant in the
        # given (all-given) axes — the branch-and-fall-back-to-`project_aux` idiom
        @test isnothing(tryproject(elt[0, 1], (Wu,)))
        @test tryproject(elt[1, 0], (Wu,)) isa AbstractTensorMap
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

        # Empty codomain: every index lands in the domain, the mirror of the flat all-codomain
        # form. The domain space type is read from the operand, so the empty codomain tuple does
        # not need to carry it.
        te = TensorAlgebra.permutedims(t, (), (3, 1, 2))
        @test space(te) == (one(A1) ← (B ⊗ dual(A1) ⊗ dual(A2)))
        @test convert(Array, te) == ref
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

    @testset "data / datatype (storage vector)" begin
        W = Rep[U₁](0 => 2, 1 => 1)
        X = Rep[U₁](0 => 1, 1 => 2)
        t = randn(rng, elt, W, X)
        # `data` reaches the underlying storage vector, and `datatype` is its type (`storagetype`).
        @test TensorAlgebra.data(t) === t.data
        @test TensorAlgebra.datatype(t) === storagetype(t) === typeof(t.data)
        # A lazy adjoint shares its parent's storage vector; the generic `data` recursion follows
        # `Base.parent` down to it, so no dedicated adjoint method is needed.
        @test TensorAlgebra.data(t') === t.data
        @test TensorAlgebra.datatype(t') === storagetype(t)
        # A `DiagonalTensorMap` is also a storage-owning leaf.
        d = DiagonalTensorMap(randn(rng, elt, reduceddim(W)), W)
        @test TensorAlgebra.data(d) === d.data
        @test TensorAlgebra.datatype(d) === storagetype(d)
    end

    @testset "ungrade / tr" begin
        W = Rep[U₁](0 => 2, 1 => 1)
        X = Rep[U₁](0 => 1, 1 => 2)
        # `ungrade` drops sectors and the arrow, so a space and its dual share the ungraded extent.
        @test TensorAlgebra.ungrade(W) == Base.OneTo(dim(W))
        @test TensorAlgebra.ungrade(dual(W)) == TensorAlgebra.ungrade(W)

        # `tr` over a codomain/domain bipartition matches TensorKit's native trace of the endomorphism.
        t = randn(rng, elt, W ⊗ X, W ⊗ X)
        t_before = copy(t)
        @test TensorAlgebra.tr(t, (:i, :j, :ip, :jp), (:i, :j), (:ip, :jp)) ≈
            LinearAlgebra.tr(t)

        # The identity fill routes through `MatrixAlgebra.one!`, which TensorKit answers with its
        # own `one!` rather than MatrixAlgebraKit's (that one speaks `AbstractMatrix` only).
        style = TensorAlgebra.MatricizeStyle(t)
        Id = TensorAlgebra.one(t, Val(2))
        @test Id ≈ TensorKit.id(TensorKit.domain(t))
        @test Id !== t
        @test space(Id) == space(t)
        @test t ≈ t_before
        for got in (
                TensorAlgebra.one(t, (:i, :j, :ip, :jp), (:i, :j), (:ip, :jp)),
                TensorAlgebra.one(t, (1, 2), (3, 4)),
                TensorAlgebra.one(style, t, Val(2)),
                TensorAlgebra.one(style, t, (1, 2), (3, 4)),
            )
            @test space(got) == space(t)
            @test got ≈ Id
        end
        @test t ≈ t_before
        # In-place, the matching split is TensorKit's own space, so it fills `t` through the view.
        c = copy(t)
        @test TensorAlgebra.one!(c, Val(2)) === c
        @test c ≈ Id
        c_style = copy(t)
        @test TensorAlgebra.one!(style, c_style, Val(2)) === c_style
        @test c_style ≈ Id
    end

    @testset "the matricize hooks cohere on the matching split" begin
        W = Rep[U₁](0 => 2, 1 => 1)
        X = Rep[U₁](0 => 1, 1 => 2)
        Z = Rep[U₁](0 => 1, 1 => 2)
        t = randn(rng, elt, W ⊗ X, Z)
        style = TensorAlgebra.MatricizeStyle(t)
        splits = (((1, 2), (3,)), ((1, 3), (2,)), ((1, 2, 3), ()), ((), (1, 2, 3)))
        for op in (identity, conj), (pc, pd) in splits
            # Regrouping a `TensorMap` is a permuted-add, so the matricization is the tensor
            # `permutedimsop` builds, spaces included. Under `conj` that dualizes every space.
            ref = TensorAlgebra.permutedimsop(op, t, pc, pd)
            if TensorAlgebra.is_output_view(TensorAlgebra.matricizeop, style, op, t, pc, pd)
                @test op === identity # only the identity op can hand back `t` itself
                @test TensorAlgebra.matricizeopview(style, op, t, pc, pd) === t
            end
            m_copy = TensorAlgebra.matricizeopcopy(style, op, t, pc, pd)
            @test m_copy !== t
            @test TensorAlgebra.data(m_copy) !== TensorAlgebra.data(t)
            @test space(m_copy) == space(ref)
            @test m_copy ≈ ref
        end
        # Only the split TensorKit already stores is a view. A regrouped one has to be built.
        @test TensorAlgebra.is_output_view(
            TensorAlgebra.matricizeop, style, identity, t, (1, 2), (3,)
        )
        @test !TensorAlgebra.is_output_view(
            TensorAlgebra.matricizeop, style, identity, t, (1, 3), (2,)
        )

        dest = similar(t)
        @test TensorAlgebra.unmatricize!(
            style, dest, matricize(style, t, (1, 2), (3,)), Val(2)
        ) === dest
        @test dest ≈ t
    end

    # The factorizations are generic over the matricize hooks, so a `TensorMap` reaches them
    # through the extension rather than through any dedicated method. Each factor is checked for
    # the spaces it carries as well as for reconstructing the input.
    @testset "factorizations" begin
        W = Rep[U₁](0 => 2, 1 => 1)
        X = Rep[U₁](0 => 1, 1 => 2)
        # Every charge sector of the fused codomain is at least as large as the domain's, which
        # is what the tall-block factorizations need.
        Z = Rep[U₁](0 => 1, 1 => 2)
        t = randn(rng, elt, W ⊗ X, Z)
        t_before = copy(t)
        labels_t = (:i, :j, :k)
        labels_l = (:i, :j)
        labels_r = (:k,)
        # Both halves of every two-factor form carry the bond on the inside, so one
        # reconstruction covers them all.
        reconstruct(F1, F2) = contractalign(
            labels_t, F1, (labels_l..., :q), F2, (:q, labels_r...)
        )

        @testset "$(nameof(f))" for f in (
                qr_compact, qr_full, lq_compact, lq_full, left_orth, right_orth, left_polar,
            )
            F1, F2 = f(t, labels_t, labels_l, labels_r)
            @test F1 isa AbstractTensorMap
            @test F2 isa AbstractTensorMap
            @test space(F1, 1) == space(t, 1)
            @test space(F1, 2) == space(t, 2)
            @test space(F2, 2) == space(t, 3)
            @test reconstruct(F1, F2) ≈ t
        end

        @testset "$(nameof(f))" for f in (svd_compact, svd_full)
            U, S, Vᴴ = f(t, labels_t, labels_l, labels_r)
            @test U isa AbstractTensorMap
            @test Vᴴ isa AbstractTensorMap
            US, labels_US = contract(U, (labels_l..., :u), S, (:u, :q))
            @test contractalign(labels_t, US, labels_US, Vᴴ, (:q, labels_r...)) ≈ t
        end

        @testset "svd_vals matches the compact spectrum" begin
            _, S, _ = svd_compact(t, labels_t, labels_l, labels_r)
            @test svd_vals(t, labels_t, labels_l, labels_r) ≈ TensorAlgebra.data(S)
        end

        @testset "the left null space annihilates the map" begin
            N = left_null(t, labels_t, labels_l, labels_r)
            @test N isa AbstractTensorMap
            @test norm(N' * t) < sqrt(eps(real(elt))) * norm(t)
        end

        @test t ≈ t_before # no factorization altered the input

        # `right_polar` needs a wide block in every sector, and `t`'s right null space is empty
        # because it has full column rank, so both get the transposed map.
        @testset "the wide map: right_polar and the right null space" begin
            tw = randn(rng, elt, Z, W ⊗ X)
            labels_w = (:k, :i, :j)
            P, Qw = right_polar(tw, labels_w, (:k,), (:i, :j))
            @test contractalign(labels_w, P, (:k, :q), Qw, (:q, :i, :j)) ≈ tw
            M = right_null(tw, labels_w, (:k,), (:i, :j))
            @test M isa AbstractTensorMap
            @test norm(tw * M') < sqrt(eps(real(elt))) * norm(tw)
        end

        @testset "the eigen family on an endomorphism" begin
            te = randn(rng, elt, W ⊗ X, W ⊗ X)
            labels_e = (:i, :j, :ip, :jp)
            labels_v, labels_vp = (:i, :j), (:ip, :jp)
            D, V = eig_full(te, labels_e, labels_v, labels_vp)
            teV = contractalign((:i, :j, :d), te, labels_e, V, (labels_vp..., :d))
            VD = contractalign((:i, :j, :d), V, (labels_v..., :dp), D, (:dp, :d))
            @test teV ≈ VD
            @test eig_vals(te, labels_e, labels_v, labels_vp) ≈ TensorAlgebra.data(D)

            # The Hermitian entries need a Hermitian input.
            th = te + te'
            Dh, Vh = eigh_full(th, labels_e, labels_v, labels_vp)
            thV = contractalign((:i, :j, :d), th, labels_e, Vh, (labels_vp..., :d))
            VhD = contractalign((:i, :j, :d), Vh, (labels_v..., :dp), Dh, (:dp, :d))
            @test thV ≈ VhD
            @test eigh_vals(th, labels_e, labels_v, labels_vp) ≈ TensorAlgebra.data(Dh)
        end
    end
end

@testset "dual/isdual on TensorKit spaces and sectors" begin
    V = Rep[U₁](0 => 2, 1 => 1)
    @test TensorAlgebra.isdual(V) == false
    @test TensorAlgebra.isdual(dual(V)) == true
    @test TensorAlgebra.dual(V) == dual(V)
    # A sector's dual is its conjugate.
    s = Irrep[U₁](1)
    @test TensorAlgebra.dual(s) == dual(s)
end

@testset "the codomain/domain accessors on a TensorMap" begin
    W = Rep[U₁](0 => 1, 1 => 1)
    t = zeros_map(Float64, (W, W), (W,))
    @test has_bipartition(t)
    @test TensorAlgebra.ndims_codomain(t) == 2
    @test TensorAlgebra.ndims_domain(t) == 1
end
