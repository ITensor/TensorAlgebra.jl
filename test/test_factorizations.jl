using Test: @test, @testset, @inferred
using TestExtras: @constinferred
using TensorAlgebra: contract, qr, svd, svdvals, TensorAlgebra, eig, eigvals
using MatrixAlgebraKit: truncrank
using LinearAlgebra: norm, diag

elts = (Float64, ComplexF64)

# QR Decomposition
# ----------------
@testset "Full QR ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  labels_A = (:a, :b, :c, :d)
  labels_Q = (:b, :a)
  labels_R = (:d, :c)

  Acopy = deepcopy(A)
  Q, R = @constinferred qr(A, labels_A, labels_Q, labels_R; full=true)
  @test A == Acopy # should not have altered initial array
  A′ = contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
  @test A ≈ A′
  @test size(Q, 1) * size(Q, 2) == size(Q, 3) # Q is unitary
end

@testset "Compact QR ($T)" for T in elts
  A = randn(T, 2, 3, 4, 5) # compact only makes a difference for less columns
  labels_A = (:a, :b, :c, :d)
  labels_Q = (:b, :a)
  labels_R = (:d, :c)

  Acopy = deepcopy(A)
  Q, R = @constinferred qr(A, labels_A, labels_Q, labels_R; full=false)
  @test A == Acopy # should not have altered initial array
  A′ = contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
  @test A ≈ A′
  @test size(Q, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

# LQ Decomposition
# ----------------
@testset "Full LQ ($T)" for T in elts
  A = randn(T, 2, 3, 4, 5)
  labels_A = (:a, :b, :c, :d)
  labels_Q = (:d, :c)
  labels_L = (:b, :a)

  Acopy = deepcopy(A)
  L, Q = @constinferred lq(A, labels_A, labels_L, labels_Q; full=true)
  @test A == Acopy # should not have altered initial array
  A′ = contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
  @test A ≈ A′
  @test size(Q, 1) == size(Q, 2) * size(Q, 3) # Q is unitary
end

@testset "Compact LQ ($T)" for T in elts
  A = randn(T, 2, 3, 4, 5)
  labels_A = (:a, :b, :c, :d)
  labels_Q = (:d, :c)
  labels_L = (:b, :a)

  Acopy = deepcopy(A)
  L, Q = @constinferred lq(A, labels_A, labels_L, labels_Q; full=true)
  @test A == Acopy # should not have altered initial array
  A′ = contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
  @test A ≈ A′
  @test size(Q, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4)) # Q is unitary
end

# Eigenvalue Decomposition
# ------------------------
@testset "Eigenvalue decomposition ($T)" for T in elts
  A = randn(T, 4, 3, 4, 3) # needs to be square
  labels_A = (:a, :b, :c, :d)
  labels_V = (:b, :a)
  labels_V′ = (:d, :c)

  Acopy = deepcopy(A)
  # type-unstable if `ishermitian` not set
  D, V = @constinferred eig(A, labels_A, labels_V, labels_V′; ishermitian=false)
  @test A == Acopy # should not have altered initial array
  @test eltype(D) == eltype(V) && eltype(D) <: Complex

  AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
  VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
  @test AV ≈ VD

  Dvals = @constinferred eigvals(A, labels_A, labels_V, labels_V′; ishermitian=false)
  @test Dvals ≈ diag(D)
  @test eltype(Dvals) <: Complex
end

@testset "Hermitian eigenvalue decomposition ($T)" for T in elts
  A = randn(T, 12, 12)
  A = reshape(A + A', 4, 3, 4, 3)
  labels_A = (:a, :b, :c, :d)
  labels_V = (:b, :a)
  labels_V′ = (:d, :c)

  Acopy = deepcopy(A)
  # type-unstable if `ishermitian` not set
  D, V = @constinferred eig(A, labels_A, labels_V, labels_V′; ishermitian=true)
  @test A == Acopy # should not have altered initial array
  @test eltype(D) <: Real
  @test eltype(V) == eltype(A)

  AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
  VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
  @test AV ≈ VD

  # TODO: broken type stability
  Dvals = eigvals(A, labels_A, labels_V, labels_V′; ishermitian=true)
  @test Dvals ≈ diag(D)
  @test eltype(Dvals) <: Real
end

# Singular Value Decomposition
# ----------------------------
@testset "Full SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  Acopy = deepcopy(A)
  U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full=true)
  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test A ≈ A′
  @test size(U, 1) * size(U, 2) == size(U, 3) # U is unitary
  @test size(Vᴴ, 1) == size(Vᴴ, 2) * size(Vᴴ, 3) # V is unitary
end

@testset "Compact SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  Acopy = deepcopy(A)
  U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full=false)
  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test A ≈ A′
  k = min(size(S)...)
  @test size(U, 3) == k == size(Vᴴ, 1)

  Svals = @constinferred svdvals(A, labels_A, labels_U, labels_Vᴴ)
  @test Svals ≈ diag(S)
end

@testset "Truncated SVD ($T)" for T in elts
  A = randn(T, 5, 4, 3, 2)
  labels_A = (:a, :b, :c, :d)
  labels_U = (:b, :a)
  labels_Vᴴ = (:d, :c)

  # test truncated SVD
  Acopy = deepcopy(A)
  _, S_untrunc, _ = svd(A, labels_A, labels_U, labels_Vᴴ)

  trunc = truncrank(size(S_untrunc, 1) - 1)
  U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; trunc)

  @test A == Acopy # should not have altered initial array
  US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
  A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
  @test norm(A - A′) ≈ S_untrunc[end]
  @test size(S, 1) == size(S_untrunc, 1) - 1
end
