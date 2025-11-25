using LinearAlgebra: LinearAlgebra, norm, diag
using MatrixAlgebraKit: truncrank
using TensorAlgebra:
    contract,
    eigen,
    eigvals,
    factorize,
    left_null,
    left_orth,
    left_polar,
    lq,
    orth,
    polar,
    qr,
    right_null,
    right_orth,
    right_polar,
    svd,
    svdvals
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

    Acopy = deepcopy(A)
    Q, R = @constinferred qr(A, labels_A, labels_Q, labels_R; full = true)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
    @test A ≈ A′
    @test size(Q, 1) * size(Q, 2) == size(Q, 3) # Q is unitary

    Q, R = qr(A, (2, 1), (4, 3); full = true)
    @test A ≈ contract(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))

    Q, R = qr(A, Val(2), Val(2); full = true)
    @test A ≈ contract((:a, :b, :c, :d), Q, (:a, :b, :q), R, (:q, :c, :d))
end

@testset "Compact QR ($T)" for T in elts
    A = randn(T, 2, 3, 4, 5) # compact only makes a difference for less columns
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:b, :a)
    labels_R = (:d, :c)

    Acopy = deepcopy(A)
    Q, R = @constinferred qr(A, labels_A, labels_Q, labels_R; full = false)
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
    L, Q = @constinferred lq(A, labels_A, labels_L, labels_Q; full = true)
    @test A == Acopy # should not have altered initial array
    A′ = contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
    @test A ≈ A′
    @test size(Q, 1) == size(Q, 2) * size(Q, 3) # Q is unitary

    L, Q = lq(A, (2, 1), (4, 3); full = true)
    @test A ≈ contract(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
end

@testset "Compact LQ ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2) # compact only makes a difference for less rows
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:d, :c)
    labels_L = (:b, :a)

    Acopy = deepcopy(A)
    L, Q = @constinferred lq(A, labels_A, labels_L, labels_Q; full = false)
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
    # type-unstable because of `ishermitian` difference
    D, V = eigen(A, labels_A, labels_V, labels_V′; ishermitian = false)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) == eltype(V) && eltype(D) <: Complex

    AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    # type-unstable because of `ishermitian` difference
    Dvals = eigvals(A, labels_A, labels_V, labels_V′; ishermitian = false)
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
    # type-unstable because of `ishermitian` difference
    D, V = eigen(A, labels_A, labels_V, labels_V′; ishermitian = true)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) <: Real
    @test eltype(V) == eltype(A)

    AV = contract((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contract((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    # type-unstable because of `ishermitian` difference
    Dvals = eigvals(A, labels_A, labels_V, labels_V′; ishermitian = true)
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
    U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full = true)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    @test size(U, 1) * size(U, 2) == size(U, 3) # U is unitary
    @test size(Vᴴ, 1) == size(Vᴴ, 2) * size(Vᴴ, 3) # V is unitary

    U, S, Vᴴ = svd(A, (2, 1), (4, 3); full = true)
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    @test A ≈ contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))

    U, S, Vᴴ = @constinferred svd(A, labels_A, labels_A, (); full = true)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(Vᴴ, 1) == 1

    U, S, Vᴴ = @constinferred svd(A, labels_A, (), labels_A; full = true)
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

    Acopy = deepcopy(A)
    U, S, Vᴴ = @constinferred svd(A, labels_A, labels_U, labels_Vᴴ; full = false)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    k = min(size(S)...)
    @test size(U, 3) == k == size(Vᴴ, 1)

    Svals = @constinferred svdvals(A, labels_A, labels_U, labels_Vᴴ)
    @test Svals ≈ diag(S)

    U, S, Vᴴ = @constinferred svd(A, labels_A, labels_A, (); full = false)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contract(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(U, ndims(U)) == 1 == size(Vᴴ, 1)

    U, S, Vᴴ = @constinferred svd(A, labels_A, (), labels_A; full = false)
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

@testset "Nullspace ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_codomain = (:b, :a)
    labels_domain = (:d, :c)

    Acopy = deepcopy(A)
    N = @constinferred left_null(A, labels_A, labels_codomain, labels_domain)
    @test A == Acopy # should not have altered initial array
    # N^ba_n' * A^ba_dc = 0
    NA = contract((:n, labels_domain...), conj(N), (labels_codomain..., :n), A, labels_A)
    @test norm(NA) ≈ 0 atol = 1.0e-14
    NN = contract((:n, :n′), conj(N), (labels_codomain..., :n), N, (labels_codomain..., :n′))
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

    Acopy = deepcopy(A)
    for (W, P) in (
            left_polar(A, labels_A, labels_W, labels_P),
            polar(A, labels_A, labels_W, labels_P; side = :left),
            polar(A, labels_A, labels_W, labels_P),
        )
        @test A == Acopy # should not have altered initial array
        A′ = contract(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
        @test A ≈ A′
        @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
    end
end

@testset "Right polar ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = deepcopy(A)
    for (P, W) in (
            right_polar(A, labels_A, labels_P, labels_W),
            polar(A, labels_A, labels_P, labels_W; side = :right),
        )
        @test A == Acopy # should not have altered initial array
        A′ = contract(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
        @test A ≈ A′
        @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
    end
end

@testset "Left orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_W = (:b, :a)
    labels_P = (:d, :c)

    Acopy = deepcopy(A)
    for (W, P) in (
            left_orth(A, labels_A, labels_W, labels_P),
            orth(A, labels_A, labels_W, labels_P; side = :left),
            orth(A, labels_A, labels_W, labels_P),
        )
        @test A == Acopy # should not have altered initial array
        A′ = contract(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
        @test A ≈ A′
        @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
    end
end

@testset "Right orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = deepcopy(A)
    for (P, W) in (
            right_orth(A, labels_A, labels_P, labels_W),
            orth(A, labels_A, labels_P, labels_W; side = :right),
        )
        @test A == Acopy # should not have altered initial array
        A′ = contract(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
        @test A ≈ A′
        @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
    end
end

@testset "factorize ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_X = (:b, :a)
    labels_Y = (:d, :c)

    Acopy = deepcopy(A)
    for orth in (:left, :right)
        X, Y = factorize(A, labels_A, labels_X, labels_Y; orth)
        @test A == Acopy # should not have altered initial array
        A′ = contract(labels_A, X, (labels_X..., :x), Y, (:x, labels_Y...))
        @test A ≈ A′
        @test size(X, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))

        X, Y = factorize(A, (2, 1), (4, 3); orth)
        @test A ≈ contract(labels_A, X, (labels_X..., :x), Y, (:x, labels_Y...))
    end
end
