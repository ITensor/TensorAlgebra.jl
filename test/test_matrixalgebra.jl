using LinearAlgebra: Diagonal, I, diag, isposdef, norm
using MatrixAlgebraKit: qr_compact, svd_trunc, truncrank
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra: MatrixAlgebra, truncdegen
using Test: @test, @testset

elts = (Float32, Float64, ComplexF32, ComplexF64)

@testset "TensorAlgebra.MatrixAlgebra (elt=$elt)" for elt in elts
    @testset "Factorizations" begin
        rng = StableRNG(123)
        A = randn(rng, elt, 3, 2)
        for positive in (false, true)
            for (Q, R) in
                (MatrixAlgebra.qr(A; positive), MatrixAlgebra.qr(A; full = false, positive))
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
            Q, R = MatrixAlgebra.qr(A; full = true, positive)
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
            for (L, Q) in
                (MatrixAlgebra.lq(A; positive), MatrixAlgebra.lq(A; full = false, positive))
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
            L, Q = MatrixAlgebra.lq(A; full = true, positive)
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
        for (W, C) in (MatrixAlgebra.orth(A), MatrixAlgebra.orth(A; side = :left))
            @test A ≈ W * C
            @test size(W) == size(A)
            @test size(C) == (size(A, 2), size(A, 2))
            @test W' * W ≈ I
            @test W * W' ≉ I
        end

        A = randn(elt, 2, 3)
        C, W = MatrixAlgebra.orth(A; side = :right)
        @test A ≈ C * W
        @test size(C) == (size(A, 1), size(A, 1))
        @test size(W) == size(A)
        @test W * W' ≈ I
        @test W' * W ≉ I

        A = randn(elt, 3, 2)
        for (W, P) in (MatrixAlgebra.polar(A), MatrixAlgebra.polar(A; side = :left))
            @test A ≈ W * P
            @test size(W) == size(A)
            @test size(P) == (size(A, 2), size(A, 2))
            @test W' * W ≈ I
            @test W * W' ≉ I
            @test isposdef(P)
        end

        A = randn(elt, 2, 3)
        P, W = MatrixAlgebra.polar(A; side = :right)
        @test A ≈ P * W
        @test size(P) == (size(A, 1), size(A, 1))
        @test size(W) == size(A)
        @test W * W' ≈ I
        @test W' * W ≉ I
        @test isposdef(P)

        A = randn(elt, 3, 2)
        for (W, C) in (MatrixAlgebra.factorize(A), MatrixAlgebra.factorize(A; orth = :left))
            @test A ≈ W * C
            @test size(W) == size(A)
            @test size(C) == (size(A, 2), size(A, 2))
            @test W' * W ≈ I
            @test W * W' ≉ I
        end

        A = randn(elt, 2, 3)
        C, W = MatrixAlgebra.factorize(A; orth = :right)
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
        for (U, S, V) in (MatrixAlgebra.svd(A), MatrixAlgebra.svd(A; full = false))
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
        U, S, V = MatrixAlgebra.svd(A; full = true)
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
    @testset "Truncate degenerate" begin
        s = Diagonal(real(elt)[2.0, 0.32, 0.3, 0.29, 0.01, 0.01])
        n = length(diag(s))
        rng = StableRNG(123)
        u, _ = qr_compact(randn(rng, elt, n, n); positive = true)
        v, _ = qr_compact(randn(rng, elt, n, n); positive = true)
        a = u * s * v

        ũ, s̃, ṽ = svd_trunc(a; trunc = truncdegen(truncrank(n); atol = 0.1))
        @test size(ũ) == (n, n)
        @test size(s̃) == (n, n)
        @test size(ṽ) == (n, n)
        @test ũ * s̃ * ṽ ≈ a

        for kwargs in (
                (; atol = eps(real(elt))),
                (; rtol = (√eps(real(elt)))),
                (; atol = eps(real(elt)), rtol = (√eps(real(elt)))),
            )
            ũ, s̃, ṽ = svd_trunc(a; trunc = truncdegen(truncrank(5); kwargs...))
            @test size(ũ) == (n, 4)
            @test size(s̃) == (4, 4)
            @test size(ṽ) == (4, n)
            @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.01, 0.01])
        end

        for kwargs in (
                (; atol = eps(real(elt))),
                (; rtol = eps(real(elt))),
                (; atol = eps(real(elt)), rtol = eps(real(elt))),
            )
            ũ, s̃, ṽ = svd_trunc(a; trunc = truncdegen(truncrank(4); kwargs...))
            @test size(ũ) == (n, 4)
            @test size(s̃) == (4, 4)
            @test size(ṽ) == (4, n)
            @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.01, 0.01])
        end

        trunc = truncdegen(truncrank(3); atol = 0.01 - √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 3)
        @test size(s̃) == (3, 3)
        @test size(ṽ) == (3, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); rtol = 0.01 / 0.3 - √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 3)
        @test size(s̃) == (3, 3)
        @test size(ṽ) == (3, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = 0.01 + √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 2)
        @test size(s̃) == (2, 2)
        @test size(ṽ) == (2, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); rtol = 0.01 / 0.29 + √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 2)
        @test size(s̃) == (2, 2)
        @test size(ṽ) == (2, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = 0.02 - √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 2)
        @test size(s̃) == (2, 2)
        @test size(ṽ) == (2, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); rtol = 0.02 / 0.29 - √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 2)
        @test size(s̃) == (2, 2)
        @test size(ṽ) == (2, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = 0.03 + √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); rtol = 0.03 / 0.29 + √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = 0.01, rtol = 0.03 / 0.29 + √eps(real(elt)))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = 0.03 + √eps(real(elt)), rtol = 0.01 / 0.29)
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = (2 - 0.29) - √(eps(real(elt))))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); rtol = (2 - 0.29) / 0.29 - √(eps(real(elt))))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 1)
        @test size(s̃) == (1, 1)
        @test size(ṽ) == (1, n)
        @test norm(ũ * s̃ * ṽ - a) ≈ norm([0.32, 0.3, 0.29, 0.01, 0.01])

        trunc = truncdegen(truncrank(3); atol = (2 - 0.29) + √(eps(real(elt))))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 0)
        @test size(s̃) == (0, 0)
        @test size(ṽ) == (0, n)
        @test norm(ũ * s̃ * ṽ) ≈ 0

        trunc = truncdegen(truncrank(3); rtol = (2 - 0.29) / 0.29 + √(eps(real(elt))))
        ũ, s̃, ṽ = svd_trunc(a; trunc)
        @test size(ũ) == (n, 0)
        @test size(s̃) == (0, 0)
        @test size(ṽ) == (0, n)
        @test norm(ũ * s̃ * ṽ) ≈ 0
    end

    @testset "gram_eigh_full" begin
        n = 5
        # Full-rank Hermitian PSD. Use a tall random factor so `B' * B`
        # is comfortably full rank even at Float32 precision (a square
        # random `B` can produce a `B' * B` whose smallest eigenvalue
        # falls below the default rtol clamp on some seeds).
        rng = StableRNG(123)
        B = randn(rng, elt, 2n, n)
        A = B' * B
        X = MatrixAlgebra.gram_eigh_full(A)
        @test X' * X ≈ A
        @test size(X) == (n, n)

        X2, Y2 = MatrixAlgebra.gram_eigh_full_with_pinv(A)
        @test X2' * X2 ≈ A
        @test X2 * Y2 ≈ I(n)

        # `!!` variant accepts a destroyable copy.
        Xb = MatrixAlgebra.gram_eigh_full!!(copy(A))
        @test Xb' * Xb ≈ A

        # Rank deficient: A is n×n of rank k < n. Recovery of A still holds;
        # X * Y is the projector onto the rank-k subspace (idempotent,
        # rank k), and P * X ≈ X (Moore–Penrose).
        k = 3
        Brd = randn(rng, elt, k, n)
        Ard = Brd' * Brd
        Xrd, Yrd = MatrixAlgebra.gram_eigh_full_with_pinv(
            Ard; rtol = sqrt(eps(real(elt)))
        )
        @test Xrd' * Xrd ≈ Ard
        P = Xrd * Yrd
        @test P * P ≈ P
        @test P * Xrd ≈ Xrd
    end

    @testset "powh_safe / sqrth_safe / invsqrth_safe" begin
        n = 4
        rng = StableRNG(123)
        B = randn(rng, elt, 2n, n)
        A = B' * B
        sqrtA = MatrixAlgebra.sqrth_safe(A)
        @test sqrtA * sqrtA ≈ A
        @test sqrtA ≈ sqrtA'

        invsqrtA = MatrixAlgebra.invsqrth_safe(A)
        @test invsqrtA * sqrtA ≈ I(n)

        # Integer power: passes through without clamping affecting result.
        @test MatrixAlgebra.powh_safe(A, 2) ≈ A * A

        # Diagonal-input dispatch path.
        D = Diagonal(rand(real(elt), n))
        @test MatrixAlgebra.sqrth_safe(D) ≈
            MatrixAlgebra.pow_diag_safe(D, 1 // 2)

        # `isdiag` fast-path: a dense matrix that happens to be diagonal-
        # structured goes through `pow_diag_safe`, not `eigh_full`, and
        # the result is still a `Diagonal` (from `MAK.diagonal`).
        Mdiag = Matrix(D)
        sqrtMdiag = MatrixAlgebra.powh_safe(Mdiag, 1 // 2)
        @test sqrtMdiag isa Diagonal
        @test sqrtMdiag ≈ MatrixAlgebra.pow_diag_safe(D, 1 // 2)

        # `pow_diag_safe` directly on a diagonal-structured `AbstractMatrix`.
        @test MatrixAlgebra.pow_diag_safe(Mdiag, 1 // 2) ≈
            MatrixAlgebra.pow_diag_safe(D, 1 // 2)

        # `pow_diag_safe` rejects non-diagonal inputs.
        @test_throws ArgumentError MatrixAlgebra.pow_diag_safe(A, 1 // 2)
    end
end
