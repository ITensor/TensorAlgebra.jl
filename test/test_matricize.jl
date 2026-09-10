using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, ReshapeMatricize, ismatricizeview, matricize,
    matricizecopy, matricizeopperm, matricizeperm, matricizeview
using Test: @test, @test_throws, @testset

# A non-`ReshapeMatricize` style, to check the always-safe generic fallback.
struct DummyMatricize <: TensorAlgebra.MatricizeStyle end

# Ground-truth matricization: permute into `(codomain..., domain...)` order, then reshape.
function matricize_ref(a, perm_codomain, perm_domain)
    a_perm = permutedims(a, (perm_codomain..., perm_domain...))
    nrow = prod(i -> size(a, i), perm_codomain; init = 1)
    ncol = prod(i -> size(a, i), perm_domain; init = 1)
    return reshape(a_perm, (nrow, ncol))
end

@testset "maybe-view matricizeperm (eltype=$elt)" for elt in (Float64, ComplexF64)
    a = randn(StableRNG(123), elt, 2, 3, 4)

    # Identity bipermutation: correct values and a view aliasing `a`.
    m = matricizeperm(a, (1,), (2, 3))
    @test m ≈ matricize_ref(a, (1,), (2, 3))
    @test Base.mightalias(m, a)

    # Every other bipermutation is a fresh permuted copy in matricized layout (no lazy
    # wrappers), including the codomain/domain swap.
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        m = matricizeperm(a, pc, pd)
        @test m ≈ matricize_ref(a, pc, pd)
        @test m isa Matrix
        @test !Base.mightalias(m, a)
    end
    @test_throws ArgumentError matricizeperm(a, (1,), (2,))

    # `conj` cannot ride a view, so it copies even on the identity bipermutation.
    m = matricizeopperm(conj, a, (1,), (2, 3))
    @test m ≈ conj.(matricize_ref(a, (1,), (2, 3)))
    @test !Base.mightalias(m, a)
    m = matricizeopperm(conj, a, (2, 3), (1,))
    @test m ≈ conj.(matricize_ref(a, (2, 3), (1,)))
    @test !Base.mightalias(m, a)
end

@testset "ismatricizeview" begin
    a = randn(StableRNG(321), 2, 3, 4)
    style = ReshapeMatricize()

    # A dense reshape matricization shares memory at every trivial split.
    @test ismatricizeview(style, a, Val(1))
    @test ismatricizeview(style, a, (1,), (2, 3))

    # The bipermutation form declares sharing only at the identity: a swap or interleaving
    # bipermutation routes through the consumers' gather branches.
    @test !ismatricizeview(style, a, (2, 3), (1,))
    @test !ismatricizeview(style, a, (3, 1), (2,))

    # A generic style declares nothing (fail-safe default).
    @test !ismatricizeview(DummyMatricize(), a, Val(1))
    @test !ismatricizeview(DummyMatricize(), a, (1,), (2, 3))

    # Writes to the shared matricization are writes to `a`.
    m = matricizeview(style, a, Val(1))
    @test m == matricize_ref(a, (1,), (2, 3))
    m[1, 1] = 42
    @test a[1, 1, 1] == 42
end

@testset "ismatricizeview coherence" begin
    rng = StableRNG(11)
    a = randn(rng, 2, 3, 4)
    style = ReshapeMatricize()

    # A declared share means `matricizeview` (and so `matricize`) aliases `a`, while
    # `matricizecopy` never does.
    for K in 0:3
        if ismatricizeview(style, a, Val(K))
            m = matricizeview(style, a, Val(K))
            @test Base.mightalias(m, a)
            @test matricize(style, a, Val(K)) == m
        end
        @test !Base.mightalias(matricizecopy(style, a, Val(K)), a)

        # The perm form of the copy leaf is owned too, and at the trivial bipermutation it
        # matches the `Val` form.
        pc = ntuple(identity, K)
        pd = ntuple(i -> K + i, 3 - K)
        m_perm = matricizecopy(style, a, pc, pd)
        @test !Base.mightalias(m_perm, a)
        @test m_perm == matricizecopy(style, a, Val(K))
    end
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        m = matricizecopy(style, a, pc, pd)
        @test m ≈ matricize_ref(a, pc, pd)
        @test !Base.mightalias(m, a)
    end
    @test_throws ArgumentError matricizecopy(style, a, (1,), (2,))

    # Both destination branches of a consumer (`contractadd!`) behave: the shared-view route
    # for the identity destination bipermutation and the gather/scatter route otherwise.
    a1 = randn(rng, 2, 3, 5)
    a2 = randn(rng, 5, 3, 2)
    ref = TensorAlgebra.contract((:i, :j, :k, :l), a1, (:i, :j, :m), a2, (:m, :k, :l))
    for labels in ((:i, :j, :k, :l), (:k, :l, :i, :j), (:k, :i, :l, :j))
        perm = map(l -> findfirst(==(l), (:i, :j, :k, :l)), labels)
        dest = randn(rng, map(d -> size(ref, d), perm)...)
        expected = permutedims(ref, perm) .+ 2.0 .* dest
        TensorAlgebra.contractadd!(
            dest,
            labels,
            a1,
            (:i, :j, :m),
            a2,
            (:m, :k, :l),
            1.0,
            2.0
        )
        @test dest ≈ expected
    end
end

@testset "view branch tracks source mutations, copy branch does not" begin
    rng = StableRNG(7)

    # Identity-bipermutation view tracks an in-place update of `a`.
    a = randn(rng, 2, 3, 4)
    m = matricizeperm(a, (1,), (2, 3))
    a .= randn(rng, 2, 3, 4)
    @test m ≈ matricize_ref(a, (1,), (2, 3))

    # Permuted copies are independent of later updates to `a`.
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        a = randn(rng, 2, 3, 4)
        m = matricizeperm(a, pc, pd)
        snapshot = copy(m)
        a .= a .+ 1
        @test m == snapshot
    end
end
