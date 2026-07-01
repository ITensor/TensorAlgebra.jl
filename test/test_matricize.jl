using LinearAlgebra: Transpose
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, PermuteMatricizeKind, ReshapeFusion,
    ReshapeMatricizeKind, TransposeMatricizeKind, matricizeopperm, matricizeperm
using Test: @test, @testset

# A non-`ReshapeFusion` style, to check the always-safe generic fallback.
struct DummyFusion <: TensorAlgebra.FusionStyle end

# Ground-truth matricization: permute into `(codomain..., domain...)` order, then reshape.
function matricize_ref(a, perm_codomain, perm_domain)
    a_perm = permutedims(a, (perm_codomain..., perm_domain...))
    nrow = prod(i -> size(a, i), perm_codomain; init = 1)
    ncol = prod(i -> size(a, i), perm_domain; init = 1)
    return reshape(a_perm, (nrow, ncol))
end

@testset "matricizekind classifier" begin
    style = ReshapeFusion()
    # Already in storage order → plain reshape view.
    @test TensorAlgebra.matricizekind(style, (1,), (2, 3)) == ReshapeMatricizeKind
    @test TensorAlgebra.matricizekind(style, (1, 2), (3,)) == ReshapeMatricizeKind
    @test TensorAlgebra.matricizekind(style, (1, 2, 3), ()) == ReshapeMatricizeKind
    @test TensorAlgebra.matricizekind(style, (), (1, 2, 3)) == ReshapeMatricizeKind
    # Pure codomain/domain swap → transpose of a reshape view.
    @test TensorAlgebra.matricizekind(style, (2, 3), (1,)) == TransposeMatricizeKind
    @test TensorAlgebra.matricizekind(style, (3,), (1, 2)) == TransposeMatricizeKind
    # Interleaved → permuted copy.
    @test TensorAlgebra.matricizekind(style, (3, 1), (2,)) == PermuteMatricizeKind
    @test TensorAlgebra.matricizekind(style, (2,), (1, 3)) == PermuteMatricizeKind
    @test TensorAlgebra.matricizekind(style, (1, 3), (2,)) == PermuteMatricizeKind
    # Generic fusion styles recognize the always-safe reshape (no-op permute) but never
    # claim a transpose (which only styles with a lazy `transpose` can realize).
    @test TensorAlgebra.matricizekind(DummyFusion(), (1,), (2, 3)) == ReshapeMatricizeKind
    @test TensorAlgebra.matricizekind(DummyFusion(), (1, 2, 3), ()) == ReshapeMatricizeKind
    @test TensorAlgebra.matricizekind(DummyFusion(), (2, 3), (1,)) == PermuteMatricizeKind
    @test TensorAlgebra.matricizekind(DummyFusion(), (3, 1), (2,)) == PermuteMatricizeKind
end

@testset "maybe-view matricizeopperm (eltype=$elt)" for elt in (Float64, ComplexF64)
    a = randn(StableRNG(123), elt, 2, 3, 4)

    # Reshape branch: correct values and a view aliasing `a`.
    m = matricizeperm(a, (1,), (2, 3))
    @test m ≈ matricize_ref(a, (1,), (2, 3))
    @test Base.mightalias(m, a)

    # Transpose branch: correct values and a transpose view aliasing `a`.
    m = matricizeperm(a, (2, 3), (1,))
    @test m ≈ matricize_ref(a, (2, 3), (1,))
    @test m isa Transpose
    @test Base.mightalias(m, a)

    # Permute branch: correct values, but a fresh copy (no aliasing).
    m = matricizeperm(a, (3, 1), (2,))
    @test m ≈ matricize_ref(a, (3, 1), (2,))
    @test !Base.mightalias(m, a)

    # `conj` cannot ride a view, so it copies even on the reshape/transpose patterns.
    m = matricizeopperm(conj, a, (1,), (2, 3))
    @test m ≈ conj.(matricize_ref(a, (1,), (2, 3)))
    @test !Base.mightalias(m, a)
    m = matricizeopperm(conj, a, (2, 3), (1,))
    @test m ≈ conj.(matricize_ref(a, (2, 3), (1,)))
    @test !Base.mightalias(m, a)
end

@testset "view branches track source mutations, copy branch does not" begin
    rng = StableRNG(7)

    # Reshape view tracks an in-place update of `a`.
    a = randn(rng, 2, 3, 4)
    m = matricizeperm(a, (1,), (2, 3))
    a .= randn(rng, 2, 3, 4)
    @test m ≈ matricize_ref(a, (1,), (2, 3))

    # Transpose view tracks an in-place update of `a`.
    a = randn(rng, 2, 3, 4)
    m = matricizeperm(a, (2, 3), (1,))
    a .= randn(rng, 2, 3, 4)
    @test m ≈ matricize_ref(a, (2, 3), (1,))

    # Permute copy is independent of later updates to `a`.
    a = randn(rng, 2, 3, 4)
    m = matricizeperm(a, (3, 1), (2,))
    snapshot = copy(m)
    a .= a .+ 1
    @test m == snapshot
end
