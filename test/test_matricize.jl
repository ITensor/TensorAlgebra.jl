using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, ReshapeMatricize, is_output_view, matricize,
    matricizeop, matricizeop!, matricizeopcopy, matricizeopview
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

@testset "maybe-view matricize (eltype=$elt)" for elt in (Float64, ComplexF64)
    a = randn(StableRNG(123), elt, 2, 3, 4)

    # Identity bipermutation: correct values and a view aliasing `a`.
    m = matricize(a, (1,), (2, 3))
    @test m ≈ matricize_ref(a, (1,), (2, 3))
    @test Base.mightalias(m, a)

    # Every other bipermutation is a fresh permuted copy in matricized layout (no lazy
    # wrappers), including the codomain/domain swap.
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        m = matricize(a, pc, pd)
        @test m ≈ matricize_ref(a, pc, pd)
        @test m isa Matrix
        @test !Base.mightalias(m, a)
    end
    @test_throws ArgumentError matricize(a, (1,), (2,))

    # `conj` cannot ride a view, so it copies even on the identity bipermutation.
    m = matricizeop(conj, a, (1,), (2, 3))
    @test m ≈ conj.(matricize_ref(a, (1,), (2, 3)))
    @test !Base.mightalias(m, a)
    m = matricizeop(conj, a, (2, 3), (1,))
    @test m ≈ conj.(matricize_ref(a, (2, 3), (1,)))
    @test !Base.mightalias(m, a)
end

@testset "is_output_view" begin
    a = randn(StableRNG(321), 2, 3, 4)
    style = ReshapeMatricize()

    # A dense reshape shares memory at the identity bipermutation.
    @test is_output_view(matricizeop, style, identity, a, (1,), (2, 3))
    @test is_output_view(matricizeop, style, identity, a, (), (1, 2, 3))
    @test is_output_view(matricizeop, style, identity, a, (1, 2, 3), ())

    # Not at a swap or an interleaving, which route through the gather branch instead.
    @test !is_output_view(matricizeop, style, identity, a, (2, 3), (1,))
    @test !is_output_view(matricizeop, style, identity, a, (3, 1), (2,))

    # And never with an operation folded in, since a reshape cannot carry a `conj`.
    @test !is_output_view(matricizeop, style, conj, a, (1,), (2, 3))

    # A generic style declares nothing (fail-safe default).
    @test !is_output_view(matricizeop, DummyMatricize(), identity, a, (1,), (2, 3))

    # Writes to the shared matricization are writes to `a`.
    m = matricizeopview(style, identity, a, (1,), (2, 3))
    @test m == matricize_ref(a, (1,), (2, 3))
    m[1, 1] = 42
    @test a[1, 1, 1] == 42
end

@testset "is_output_view coherence" begin
    rng = StableRNG(11)
    a = randn(rng, 2, 3, 4)
    style = ReshapeMatricize()

    # A declared share means `matricizeopview` (and so `matricize`) aliases `a`, while
    # `matricizeopcopy` never does.
    for K in 0:3
        pc = ntuple(identity, K)
        pd = ntuple(i -> K + i, 3 - K)
        if is_output_view(matricizeop, style, identity, a, pc, pd)
            m = matricizeopview(style, identity, a, pc, pd)
            @test Base.mightalias(m, a)
            @test matricize(style, a, pc, pd) == m
        end
        m_copy = matricizeopcopy(style, identity, a, pc, pd)
        @test !Base.mightalias(m_copy, a)
        @test m_copy == matricize_ref(a, pc, pd)
    end
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        m = matricizeopcopy(style, identity, a, pc, pd)
        @test m ≈ matricize_ref(a, pc, pd)
        @test !Base.mightalias(m, a)
    end
    @test_throws ArgumentError matricizeopcopy(style, identity, a, (1,), (2,))

    # The allocation and write hooks compose into the copy form.
    for (pc, pd) in (((1,), (2, 3)), ((3, 1), (2,)))
        for op in (identity, conj)
            dest = TensorAlgebra.allocate_output(matricizeop, style, op, a, pc, pd)
            @test size(dest) == size(matricize_ref(a, pc, pd))
            matricizeop!(dest, style, op, a, pc, pd)
            @test dest ≈ op.(matricize_ref(a, pc, pd))
            @test dest ≈ matricizeopcopy(style, op, a, pc, pd)
        end
    end

    # Both destination branches of a consumer (`contractadd!`) behave: the shared-view route
    # for the identity destination bipermutation and the gather/scatter route otherwise.
    a1 = randn(rng, 2, 3, 5)
    a2 = randn(rng, 5, 3, 2)
    ref = TensorAlgebra.contractalign((:i, :j, :k, :l), a1, (:i, :j, :m), a2, (:m, :k, :l))
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
    m = matricize(a, (1,), (2, 3))
    a .= randn(rng, 2, 3, 4)
    @test m ≈ matricize_ref(a, (1,), (2, 3))

    # Permuted copies are independent of later updates to `a`.
    for (pc, pd) in (((2, 3), (1,)), ((3, 1), (2,)))
        a = randn(rng, 2, 3, 4)
        m = matricize(a, pc, pd)
        snapshot = copy(m)
        a .= a .+ 1
        @test m == snapshot
    end
end
