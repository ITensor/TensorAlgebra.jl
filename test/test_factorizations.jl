using LinearAlgebra: LinearAlgebra, I, diag, norm
using MatrixAlgebraKit: truncrank
using TensorAlgebra: TensorAlgebra, contract, eig_full, eig_vals, eigh_full, eigh_vals,
    gram_eigh_full, gram_eigh_full_with_pinv, left_null, left_orth, left_polar, lq_compact,
    lq_full, qr_compact, qr_full, right_null, right_orth, right_polar, svd_compact,
    svd_full, svd_trunc, svd_vals
using Test: @test, @testset
using TestExtras: @constinferred

elts = (Float64, ComplexF64)

# QR Decomposition
# ----------------
@testset "Full QR ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:b, :a)
    labels_R = (:d, :c)

    Acopy = copy(A)
    Q, R = @constinferred qr_full(A, labels_A, labels_Q, labels_R)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
    @test A ≈ A′
    @test size(Q, 1) * size(Q, 2) == size(Q, 3) # Q is unitary

    Q, R = qr_full(A, (2, 1), (4, 3))
    @test A ≈ contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))

    Q, R = qr_full(A, Val(2))
    @test A ≈ contract((:a, :b, :c, :d), Q, (:a, :b, :q), R, (:q, :c, :d))
end

@testset "Compact QR ($T)" for T in elts
    A = randn(T, 2, 3, 4, 5) # compact only makes a difference for less columns
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:b, :a)
    labels_R = (:d, :c)

    Acopy = copy(A)
    Q, R = @constinferred qr_compact(A, labels_A, labels_Q, labels_R)
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

    Acopy = copy(A)
    L, Q = @constinferred lq_full(A, labels_A, labels_L, labels_Q)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
    @test A ≈ A′
    @test size(Q, 1) == size(Q, 2) * size(Q, 3) # Q is unitary

    L, Q = lq_full(A, (2, 1), (4, 3))
    @test A ≈ contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
end

@testset "Compact LQ ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2) # compact only makes a difference for less rows
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:d, :c)
    labels_L = (:b, :a)

    Acopy = copy(A)
    L, Q = @constinferred lq_compact(A, labels_A, labels_L, labels_Q)
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

    Acopy = copy(A)
    D, V = eig_full(A, labels_A, labels_V, labels_V′)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) == eltype(V) && eltype(D) <: Complex

    AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    Dvals = eig_vals(A, labels_A, labels_V, labels_V′)
    @test Dvals ≈ diag(D)
    @test eltype(Dvals) <: Complex
end

@testset "Hermitian eigenvalue decomposition ($T)" for T in elts
    A = randn(T, 12, 12)
    A = reshape(A + A', 4, 3, 4, 3)
    labels_A = (:a, :b, :c, :d)
    labels_V = (:b, :a)
    labels_V′ = (:d, :c)

    Acopy = copy(A)
    D, V = eigh_full(A, labels_A, labels_V, labels_V′)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) <: Real
    @test eltype(V) == eltype(A)

    AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    Dvals = eigh_vals(A, labels_A, labels_V, labels_V′)
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

    Acopy = copy(A)
    U, S, Vᴴ = @constinferred svd_full(A, labels_A, labels_U, labels_Vᴴ)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    @test size(U, 1) * size(U, 2) == size(U, 3) # U is unitary
    @test size(Vᴴ, 1) == size(Vᴴ, 2) * size(Vᴴ, 3) # V is unitary

    U, S, Vᴴ = svd_full(A, (2, 1), (4, 3))
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    @test A ≈ contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))

    U, S, Vᴴ = @constinferred svd_full(A, labels_A, labels_A, ())
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(Vᴴ, 1) == 1

    U, S, Vᴴ = @constinferred svd_full(A, labels_A, (), labels_A)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (:u,), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_A...))
    @test A ≈ A′
    @test size(U, 2) == 1
end

@testset "Compact SVD ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_U = (:b, :a)
    labels_Vᴴ = (:d, :c)

    Acopy = copy(A)
    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, labels_U, labels_Vᴴ)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    k = min(size(S)...)
    @test size(U, 3) == k == size(Vᴴ, 1)

    Svals = @constinferred svd_vals(A, labels_A, labels_U, labels_Vᴴ)
    @test Svals ≈ diag(S)

    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, labels_A, ())
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(U, ndims(U)) == 1 == size(Vᴴ, 1)

    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, (), labels_A)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (:u,), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_A...))
    @test A ≈ A′
    @test size(U, 1) == 1 == size(Vᴴ, 1)
end

@testset "Truncated SVD ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_U = (:b, :a)
    labels_Vᴴ = (:d, :c)

    # test truncated SVD
    Acopy = copy(A)
    _, S_untrunc, _ = svd_compact(A, labels_A, labels_U, labels_Vᴴ)

    trunc = truncrank(size(S_untrunc, 1) - 1)
    U, S, Vᴴ, ϵ = @constinferred svd_trunc(A, labels_A, labels_U, labels_Vᴴ; trunc)

    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test norm(A - A′) ≈ S_untrunc[end]
    @test size(S, 1) == size(S_untrunc, 1) - 1
    # `ϵ` is the 2-norm of the discarded singular values (here the single dropped value).
    @test ϵ ≈ S_untrunc[end]
end

@testset "Nullspace ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_codomain = (:b, :a)
    labels_domain = (:d, :c)

    Acopy = copy(A)
    N = @constinferred left_null(A, labels_A, labels_codomain, labels_domain)
    @test A == Acopy # should not have altered initial array
    # N^ba_n' * A^ba_dc = 0
    NA = contract((:n, labels_domain...), conj(N), (labels_codomain..., :n), A, labels_A)
    @test norm(NA) ≈ 0 atol = 1.0e-14
    NN =
        contract((:n, :n′), conj(N), (labels_codomain..., :n), N, (labels_codomain..., :n′))
    @test NN ≈ LinearAlgebra.I

    Nᴴ = @constinferred right_null(A, labels_A, labels_codomain, labels_domain)
    @test A == Acopy # should not have altered initial array
    # A^ba_dc * N^dc_n' = 0
    AN = contract((labels_codomain..., :n), A, labels_A, conj(Nᴴ), (:n, labels_domain...))
    @test norm(AN) ≈ 0 atol = 1.0e-14
    NN = contract((:n, :n′), Nᴴ, (:n, labels_domain...), Nᴴ, (:n′, labels_domain...))
end

@testset "Left polar ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_W = (:b, :a)
    labels_P = (:d, :c)

    Acopy = copy(A)
    W, P = left_polar(A, labels_A, labels_W, labels_P)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
    @test A ≈ A′
    @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

@testset "Right polar ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = copy(A)
    P, W = right_polar(A, labels_A, labels_P, labels_W)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
    @test A ≈ A′
    @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

@testset "Left orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_W = (:b, :a)
    labels_P = (:d, :c)

    Acopy = copy(A)
    W, P = left_orth(A, labels_A, labels_W, labels_P)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
    @test A ≈ A′
    @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))

    W, P = left_orth(A, (2, 1), (4, 3))
    @test A ≈ contract(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
end

@testset "Right orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = copy(A)
    P, W = right_orth(A, labels_A, labels_P, labels_W)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
    @test A ≈ A′
    @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))

    P, W = right_orth(A, (2, 1), (4, 3))
    @test A ≈ contract(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
end

# Gram factorization
# ------------------
# Build a Hermitian positive semi-definite tensor A[a,b,c,d] with codomain
# (a, b) and domain (c, d): pick a random B[k, a, b] (k = aux), then form
# A = B' * B over k. By construction A ≈ X' * X for X[r, a, b] with rank r
# bounded by k (rank leg first, following the Cholesky `A = U' * U`
# convention).
@testset "Full-rank gram_eigh_full ($T)" for T in elts
    B = randn(T, 6, 2, 3) # k = 6, codomain = (a, b) of size 2*3 = 6 -> full rank
    A = contract((:a, :b, :c, :d), conj(B), (:k, :a, :b), B, (:k, :c, :d))
    labels_A = (:a, :b, :c, :d)
    labels_X = (:a, :b)
    labels_Y = (:c, :d)

    Acopy = copy(A)
    X = @constinferred gram_eigh_full(A, labels_A, labels_X, labels_Y)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, X, (:a, :b, :r), conj(X), (:c, :d, :r))
    @test A ≈ A′
    @test size(X, ndims(X)) == size(A, 1) * size(A, 2)

    # `Val`, perm, and label entries agree.
    @test gram_eigh_full(A, Val(2)) ≈ X
    @test gram_eigh_full(A, (1, 2), (3, 4)) ≈ X

    # `with_pinv` variant: Y is a left inverse of X (Y * X ≈ I on the
    # rank subspace).
    X2, Y2 = @constinferred gram_eigh_full_with_pinv(A, labels_A, labels_X, labels_Y)
    @test A ≈ contract(labels_A, X2, (:a, :b, :r), conj(X2), (:c, :d, :r))
    YX = contract((:r, :s), Y2, (:r, :a, :b), X2, (:a, :b, :s))
    @test YX ≈ I
end

@testset "Rank-deficient gram_eigh_full ($T)" for T in elts
    B = randn(T, 4, 2, 3) # k = 4 < codomain dim 6, so A is rank-4
    A = contract((:a, :b, :c, :d), conj(B), (:k, :a, :b), B, (:k, :c, :d))

    # Recovery of A is independent of the `rtol` cutoff because all
    # nonzero eigenvalues sit far above any reasonable threshold.
    X = gram_eigh_full(A, Val(2); rtol = 1.0e-10)
    @test A ≈ contract(
        (:a, :b, :c, :d), X, (:a, :b, :r), conj(X), (:c, :d, :r)
    )

    # Moore–Penrose-like identity: X * Y * X ≈ X when Y is pinv(X). With
    # cod-first X and rank-first Y, contract Y[r, a, b] * X[a, b, s] → P[r, s]
    # (projector onto the rank subspace), then X * P → X.
    X2, Y2 = gram_eigh_full_with_pinv(A, Val(2); rtol = 1.0e-10)
    P = contract((:r, :s), Y2, (:r, :a, :b), X2, (:a, :b, :s))
    XP = contract((:c, :d, :r), X2, (:c, :d, :s), P, (:s, :r))
    @test XP ≈ X2
end

# one (identity tensor)
# ---------------------
# An identity tensor matricized along its codomain/domain partition is the
# identity matrix.
@testset "one ($T)" for T in elts
    A = randn(T, 2, 3, 2, 3)
    labels_A = (:a, :b, :c, :d)
    labels_cod = (:a, :b)
    labels_dom = (:c, :d)

    Acopy = copy(A)
    Id = @constinferred TensorAlgebra.one(A, labels_A, labels_cod, labels_dom)
    @test A == Acopy # should not have altered initial array

    @test size(Id) == size(A)
    @test eltype(Id) === T

    @test TensorAlgebra.matricize(Id, Val(2)) ≈ I

    # `Val`, perm, and label entries agree.
    @test TensorAlgebra.one(A, Val(2)) ≈ Id
    @test TensorAlgebra.one(A, (1, 2), (3, 4)) ≈ Id

    # Non-trivial codomain/domain partition: codomain (a, b) interleaved with
    # domain (c, d) in the input layout. The result is permuted into the
    # canonical (cod, dom) order before matricizing, and the matricized form
    # is again the identity matrix.
    B = randn(T, 2, 2, 2, 2)
    labels_B = (:a, :c, :b, :d)
    Id_perm = TensorAlgebra.one(B, labels_B, labels_cod, labels_dom)
    @test TensorAlgebra.matricize(Id_perm, Val(2)) ≈ I
    # Perm- and biperm-tuple forms agree with the label form.
    @test TensorAlgebra.one(B, (1, 3), (2, 4)) ≈ Id_perm

    # In-place `one!` fills the identity into its argument and returns it.
    C = randn(T, 2, 3, 2, 3)
    Cret = @constinferred TensorAlgebra.one!(C, Val(2))
    @test Cret === C
    @test TensorAlgebra.matricize(C, Val(2)) ≈ I
    @test C ≈ TensorAlgebra.one(A, Val(2))

    # `unmatricize!` scatters a fused matrix back into an existing array.
    D = randn(T, 2, 3, 2, 3)
    Dmat = TensorAlgebra.matricize(D, Val(2))
    E = similar(D)
    Eret = TensorAlgebra.unmatricize!(E, Dmat, Val(2))
    @test Eret === E
    @test E ≈ D
end
