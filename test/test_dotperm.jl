using LinearAlgebra: dot, norm
using StableRNGs: StableRNG
using TensorAlgebra: dotperm
using TensorKit: TensorKit, FermionParity, Rep, SU₂, U₁, Vect, randn, ←, ⊗
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset

@testset "dotperm dense (eltype=$elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(123)
    a = randn(rng, elt, 2, 3, 4)
    b = randn(rng, elt, 3, 4, 2)
    @test dotperm(a, a, (1, 2, 3)) ≈ norm(a)^2
    @test dotperm(a, b, (3, 1, 2)) ≈ dot(a, permutedims(b, (3, 1, 2)))
    @test dotperm(a, b, (3, 1, 2), ()) ≈ dot(a, permutedims(b, (3, 1, 2)))
    @test dotperm(im * a, b, (3, 1, 2)) ≈ -im * dotperm(a, b, (3, 1, 2))
    @test_throws ArgumentError dotperm(a, b, (3, 1), ())
    @test_throws ArgumentError dotperm(a, b, (3, 1, 1), ())
    @test_throws ArgumentError dotperm(a, b, (3, 1), (2,))
    @test_throws DimensionMismatch dotperm(a, zeros(elt, 3, 4, 3), (3, 1, 2))
    @test dotperm(fill(elt(2)), fill(elt(3)), ()) == 6
end

@testset "dotperm TensorKit (sector=$sector, eltype=$elt)" for sector in (U₁, SU₂),
        elt in (Float64, ComplexF64)

    rng = StableRNG(456)
    V = sector === U₁ ? Rep[U₁](0 => 2, 1 => 1) : Rep[SU₂](0 => 2, 1 // 2 => 1)
    a = randn(rng, elt, V ⊗ V ← V)
    b = randn(rng, elt, V ⊗ V ← V)
    @test dotperm(a, a, (1, 2), (3,)) ≈ norm(a)^2
    @test dotperm(a, b, (1, 2), (3,)) ≈ dot(a, b)
    @test dotperm(a, b, (2, 1), (3,)) ≈ dot(a, TensorKit.permute(b, ((2, 1), (3,))))
    @test_throws ArgumentError dotperm(a, b, (1, 2, 3))
end

@testset "dotperm fermionic TensorKit rebasing" begin
    rng = StableRNG(789)
    V = Vect[FermionParity](0 => 2, 1 => 2)
    a = randn(rng, ComplexF64, V ⊗ V ← V)
    b = randn(rng, ComplexF64, V ⊗ V ← V)
    @test dotperm(a, b, (2, 1), (3,)) ≈ dot(a, TensorKit.permute(b, ((2, 1), (3,))))
    b_rebased = TensorKit.permute(b, ((1,), (2, 3)))
    @test dotperm(a, b_rebased, (1, 2), (3,)) ≈ dot(a, b)
    a_rebased = TensorKit.permute(a, ((1,), (2, 3)))
    @test dotperm(a, a_rebased, (1, 2), (3,)) ≈ norm(a)^2
end
