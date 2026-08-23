using StableRNGs: StableRNG
using TensorAlgebra:
    TensorAlgebra, ReshapeMatricize, matricizeopperm, matricizeperm, trymatricizeview
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

@testset "trymatricizeview" begin
    a = randn(StableRNG(321), 2, 3, 4)
    style = ReshapeMatricize()

    # Order-preserving splits share memory: a reshape view for the identity bipermutation,
    # a lazy transpose of it for the codomain/domain swap.
    m = trymatricizeview(style, a, (1,), (2, 3))
    @test m == matricize_ref(a, (1,), (2, 3))
    @test Base.mightalias(m, a)
    @test trymatricizeview(style, a, Val(1)) == m
    m = trymatricizeview(style, a, (2, 3), (1,))
    @test m == matricize_ref(a, (2, 3), (1,))
    @test Base.mightalias(m, a)

    # An interleaving split would move data, so no memory-sharing matricization exists.
    @test isnothing(trymatricizeview(style, a, (3, 1), (2,)))

    # A generic style declares nothing.
    @test isnothing(trymatricizeview(DummyMatricize(), a, Val(1)))
    @test isnothing(trymatricizeview(DummyMatricize(), a, (1,), (2, 3)))

    # Writes to the shared matricization are writes to `a`.
    m = trymatricizeview(style, a, (1,), (2, 3))
    m[1, 1] = 42
    @test a[1, 1, 1] == 42
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
