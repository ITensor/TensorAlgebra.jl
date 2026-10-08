using LinearAlgebra: dot, norm
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, dotperm
using TensorKit: TensorKit, FermionParity, Rep, SU₂, U₁, Vect, randn, ←, ⊗
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset

@testset "dotperm dense (eltype=$elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(123)
    a = randn(rng, elt, 2, 3, 4)
    b = randn(rng, elt, 3, 4, 2)
    @test dotperm(a, a, (1, 2, 3)) ≈ norm(a)^2
    @test dotperm(a, b, (3, 1, 2)) ≈ dot(a, permutedims(b, (3, 1, 2)))
    @test dotperm(a, b, [3, 1, 2]) ≈ dot(a, permutedims(b, (3, 1, 2)))
    @test dotperm(im * a, b, (3, 1, 2)) ≈ -im * dotperm(a, b, (3, 1, 2))
    @test_throws ArgumentError dotperm(a, b, (3, 1))
    @test_throws ArgumentError dotperm(a, b, [3, 1])
    @test_throws ArgumentError dotperm(a, b, [3, 1, 2, 4])
    @test_throws ArgumentError dotperm(a, b, (3, 1, 1))
    @test_throws ArgumentError dotperm(a, b, (3, 1, 4))
    @test_throws ArgumentError dotperm(a, vec(b), (3, 1, 2))
    @test_throws DimensionMismatch dotperm(a, zeros(elt, 3, 4, 3), (3, 1, 2))
    @test_throws DimensionMismatch dotperm(a, zeros(elt, 4, 3, 2), (3, 1, 2))
    @test dotperm(fill(elt(2)), fill(elt(3)), ()) == 6
end

@testset "dotperm TensorKit (sector=$sector, eltype=$elt)" for sector in (U₁, SU₂),
        elt in (Float64, ComplexF64)

    rng = StableRNG(456)
    V = sector === U₁ ? Rep[U₁](0 => 2, 1 => 1) : Rep[SU₂](0 => 2, 1 // 2 => 1)
    a = randn(rng, elt, V ⊗ V ← V)
    b = randn(rng, elt, V ⊗ V ← V)
    @test dotperm(a, a, (1, 2, 3)) ≈ norm(a)^2
    @test dotperm(a, b, (1, 2, 3)) ≈ dot(a, b)
    @test dotperm(a, b, (2, 1, 3)) ≈ dot(a, TensorKit.permute(b, ((2, 1), (3,))))
    @test_throws ArgumentError dotperm(a, b, (1, 2))
end

@testset "dotperm fermionic TensorKit rebasing" begin
    rng = StableRNG(789)
    V = Vect[FermionParity](0 => 2, 1 => 2)
    a = randn(rng, ComplexF64, V ⊗ V ← V)
    b = randn(rng, ComplexF64, V ⊗ V ← V)
    @test dotperm(a, b, (2, 1, 3)) ≈ dot(a, TensorKit.permute(b, ((2, 1), (3,))))
    b_rebased = TensorKit.permute(b, ((1,), (2, 3)))
    @test dotperm(a, b_rebased, (1, 2, 3)) ≈ dot(a, b)
    a_rebased = TensorKit.permute(a, ((1,), (2, 3)))
    @test dotperm(a, a_rebased, (1, 2, 3)) ≈ norm(a)^2
    for n in 0:3
        pc, pd = Tuple(1:n), Tuple((n + 1):3)
        a_split = TensorKit.permute(a, (pc, pd))
        b_split = TensorKit.permute(b, (pc, pd))
        @test dotperm(a_split, b_rebased, (1, 2, 3)) ≈ dot(a_split, b_split)
    end
    b_permuted = TensorKit.permute(b, ((2,), (1, 3)))
    @test dotperm(a, b_permuted, (2, 1, 3)) ≈ dot(a, b)
end

@testset "labeled dot" begin
    @test TensorAlgebra.dot !== dot
    rng = StableRNG(321)
    a = randn(rng, ComplexF64, 2, 3, 4)
    b = randn(rng, ComplexF64, 3, 4, 2)
    expected = dot(a, permutedims(b, (3, 1, 2)))
    @test TensorAlgebra.dot(a, (:i, :j, :k), b, (:j, :k, :i)) ≈ expected
    @test TensorAlgebra.dot(a, [10, 20, 30], b, [20, 30, 10]) ≈ expected
    @test TensorAlgebra.dot(im * a, (:i, :j, :k), b, (:j, :k, :i)) ≈ -im * expected
    @test_throws ArgumentError TensorAlgebra.dot(a, (:i, :j), b, (:j, :k, :i))
    @test_throws ArgumentError TensorAlgebra.dot(a, [10, 20], b, [20, 30, 10])
    @test_throws ArgumentError TensorAlgebra.dot(a, [10, 20, 30], b, [20, 30, 10, 40])
    @test_throws ArgumentError TensorAlgebra.dot(a, (:i, :j, :k), b, (:j, :k))
    @test_throws ArgumentError TensorAlgebra.dot(a, (:i, :i, :k), b, (:j, :k, :i))
    @test_throws ArgumentError TensorAlgebra.dot(a, (:i, :j, :k), b, (:j, :j, :i))
    @test_throws ArgumentError TensorAlgebra.dot(a, (:i, :j, :k), b, (:j, :k, :l))
    @test_throws DimensionMismatch TensorAlgebra.dot(
        a,
        (:i, :j, :k),
        zeros(ComplexF64, 4, 3, 2),
        (:j, :k, :i)
    )
    @test TensorAlgebra.dot([1 2; 3 4], (:i, :j), [5 6; 7 8], (:j, :i)) == 69
end

@testset "labeled dot fermionic TensorKit rebasing" begin
    rng = StableRNG(987)
    V = Vect[FermionParity](0 => 2, 1 => 2)
    a = randn(rng, ComplexF64, V ⊗ V ← V)
    b = randn(rng, ComplexF64, V ⊗ V ← V)
    b_rebased = TensorKit.permute(b, ((2,), (1, 3)))
    @test TensorAlgebra.dot(a, (:i, :j, :k), b_rebased, (:j, :i, :k)) ≈ dot(a, b)
    @test TensorAlgebra.dot(b_rebased, (:j, :i, :k), a, (:i, :j, :k)) ≈ conj(dot(a, b))
end
