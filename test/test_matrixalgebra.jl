using LinearAlgebra: I, diag
using TensorAlgebra.MatrixAlgebra: MatrixAlgebra
using Test: @test, @testset

elts = (Float32, Float64, ComplexF32, ComplexF64)

@testset "elt=$elt" for elt in elts
  A = randn(elt, 3, 2)
  for positive in (false, true)
    for (Q, R) in (MatrixAlgebra.qr(A; positive), MatrixAlgebra.qr(A; full=false, positive))
      @test A ≈ Q * R
      @test size(Q) == size(A)
      @test size(R) == (size(A, 2), size(A, 2))
      @test Q' * Q ≈ I
      @test Q * Q' ≉ I
      if positive
        @test all(≥(0), real(diag(R)))
        @test all(≈(0), imag(diag(R)))
      end
    end
  end

  A = randn(elt, 3, 2)
  for positive in (false, true)
    Q, R = MatrixAlgebra.qr(A; full=true, positive)
    @test A ≈ Q * R
    @test size(Q) == (size(A, 1), size(A, 1))
    @test size(R) == size(A)
    @test Q' * Q ≈ I
    @test Q * Q' ≈ I
    if positive
      @test all(≥(0), real(diag(R)))
      @test all(≈(0), imag(diag(R)))
    end
  end

  A = randn(elt, 2, 3)
  for positive in (false, true)
    for (L, Q) in (MatrixAlgebra.lq(A; positive), MatrixAlgebra.lq(A; full=false, positive))
      @test A ≈ L * Q
      @test size(L) == (size(A, 1), size(A, 1))
      @test size(Q) == size(A)
      @test Q * Q' ≈ I
      @test Q' * Q ≉ I
      if positive
        @test all(≥(0), real(diag(L)))
        @test all(≈(0), imag(diag(L)))
      end
    end
  end

  A = randn(elt, 3, 2)
  for positive in (false, true)
    L, Q = MatrixAlgebra.lq(A; full=true, positive)
    @test A ≈ L * Q
    @test size(L) == size(A)
    @test size(Q) == (size(A, 2), size(A, 2))
    @test Q * Q' ≈ I
    @test Q' * Q ≈ I
    if positive
      @test all(≥(0), real(diag(L)))
      @test all(≈(0), imag(diag(L)))
    end
  end

  A = randn(elt, 3, 2)
  for (W, C) in (MatrixAlgebra.orth(A), MatrixAlgebra.orth(A; side=:left))
    @test A ≈ W * C
    @test size(W) == size(A)
    @test size(C) == (size(A, 2), size(A, 2))
    @test W' * W ≈ I
    @test W * W' ≉ I
  end

  A = randn(elt, 2, 3)
  C, W = MatrixAlgebra.orth(A; side=:right)
  @test A ≈ C * W
  @test size(C) == (size(A, 1), size(A, 1))
  @test size(W) == size(A)
  @test W * W' ≈ I
  @test W' * W ≉ I

  A = randn(elt, 3, 2)
  for (W, P) in (MatrixAlgebra.polar(A), MatrixAlgebra.polar(A; side=:left))
    @test A ≈ W * P
    @test size(W) == size(A)
    @test size(P) == (size(A, 2), size(A, 2))
    @test W' * W ≈ I
    @test W * W' ≉ I
    @test isposdef(P)
  end

  A = randn(elt, 2, 3)
  P, W = MatrixAlgebra.polar(A; side=:right)
  @test A ≈ P * W
  @test size(P) == (size(A, 1), size(A, 1))
  @test size(W) == size(A)
  @test W * W' ≈ I
  @test W' * W ≉ I
  @test isposdef(P)

  A = randn(elt, 3, 2)
  for (W, C) in (MatrixAlgebra.factorize(A), MatrixAlgebra.factorize(A; orth=:left))
    @test A ≈ W * C
    @test size(W) == size(A)
    @test size(C) == (size(A, 2), size(A, 2))
    @test W' * W ≈ I
    @test W * W' ≉ I
  end

  A = randn(elt, 2, 3)
  C, W = MatrixAlgebra.factorize(A; orth=:right)
  @test A ≈ C * W
  @test size(C) == (size(A, 1), size(A, 1))
  @test size(W) == size(A)
  @test W * W' ≈ I
  @test W' * W ≉ I

  A = randn(elt, 3, 3)
  D, V = MatrixAlgebra.eigen(A)
  @test A * V ≈ V * D
  @test MatrixAlgebra.eigvals(A) ≈ diag(D)

  A = randn(elt, 3, 2)
  for (U, S, V) in (MatrixAlgebra.svd(A), MatrixAlgebra.svd(A; full=false))
    @test A ≈ U * S * V
    @test size(U) == size(A)
    @test size(S) == (size(A, 2), size(A, 2))
    @test size(V) == (size(A, 2), size(A, 2))
    @test U' * U ≈ I
    @test U * U' ≉ I
    @test V * V' ≈ I
    @test V' * V ≈ I
    @test MatrixAlgebra.svdvals(A) ≈ diag(S)
  end

  A = randn(elt, 3, 2)
  U, S, V = MatrixAlgebra.svd(A; full=true)
  @test A ≈ U * S * V
  @test size(U) == (size(A, 1), size(A, 1))
  @test size(S) == size(A)
  @test size(V) == (size(A, 2), size(A, 2))
  @test U' * U ≈ I
  @test U * U' ≈ I
  @test V * V' ≈ I
  @test V' * V ≈ I
  @test MatrixAlgebra.svdvals(A) ≈ diag(S)
end
