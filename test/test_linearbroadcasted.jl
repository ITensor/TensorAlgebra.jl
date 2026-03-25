using Base.Broadcast: Broadcast as BC
using TensorAlgebra: TensorAlgebra as TA, linearbroadcasted
using Test: @test, @test_throws, @testset

@testset "LinearBroadcasted and Mul" begin
    @testset "construction and materialization" begin
        a = randn(ComplexF64, 3, 3)
        b = randn(ComplexF64, 3, 3)
        c = randn(ComplexF64, 3, 3)

        x = linearbroadcasted(*, 2, a)
        @test x ≡ TA.ScaledBroadcasted(2, a)
        @test copy(x) ≈ 2a

        x = linearbroadcasted(conj, a)
        @test x ≡ TA.ConjBroadcasted(a)
        @test copy(x) ≈ conj(a)

        x = linearbroadcasted(+, a, b)
        @test x ≡ TA.AddBroadcasted(a, b)
        @test copy(x) ≈ a + b

        x = TA.Mul(a, b)
        @test copy(x) ≈ a * b

        x = linearbroadcasted(+, TA.Mul(a, b), c)
        @test x ≡ TA.AddBroadcasted(TA.Mul(a, b), c)
        @test copy(x) ≈ a * b + c

        x = linearbroadcasted(
            +,
            linearbroadcasted(*, 2, TA.Mul(a, b)),
            linearbroadcasted(*, 3, c)
        )
        @test x ≡ TA.AddBroadcasted(
            TA.ScaledBroadcasted(2, TA.Mul(a, b)), TA.ScaledBroadcasted(3, c)
        )
        @test copy(x) ≈ 2 * a * b + 3 * c
    end
    @testset "linear broadcast lowering" begin
        a = randn(ComplexF64, 2, 2)
        style = BC.DefaultArrayStyle{2}()

        @test TA.broadcasted_linear(identity, a) ≡ a
        @test TA.broadcasted_linear(Base.Fix1(*, 2), a) ≡ linearbroadcasted(*, 2, a)
        @test TA.broadcasted_linear(Base.Fix2(*, 2), a) ≡ linearbroadcasted(*, a, 2)
        @test TA.broadcasted_linear(Base.Fix2(/, 2), a) ≡ linearbroadcasted(/, a, 2)
        @test TA.broadcasted_linear(style, identity, a) ≡ a
        @test TA.broadcasted_linear(style, Base.Fix1(*, 2), a) ≡
            linearbroadcasted(*, 2, a)
        @test TA.broadcasted_linear(style, Base.Fix2(*, 2), a) ≡
            linearbroadcasted(*, a, 2)
        @test TA.broadcasted_linear(style, Base.Fix2(/, 2), a) ≡
            linearbroadcasted(/, a, 2)
        @test TA.broadcasted_linear(style, conj, a) ≡ linearbroadcasted(conj, a)
        @test_throws ArgumentError TA.broadcasted_linear(style, exp, a)
    end
    @testset "linearbroadcasted algebra" begin
        a = randn(ComplexF64, 3, 3)

        # Scaling absorbs coefficients
        @test linearbroadcasted(*, 3, linearbroadcasted(*, 2, a)) ≡
            TA.ScaledBroadcasted(6, a)

        # Conjugation of scaled
        x = linearbroadcasted(conj, linearbroadcasted(*, 2im, a))
        @test x ≡ TA.ScaledBroadcasted(-2im, TA.ConjBroadcasted(a))

        # Double conjugation cancels
        @test linearbroadcasted(conj, linearbroadcasted(conj, a)) ≡ a

        # Subtraction
        b = randn(ComplexF64, 3, 3)
        x = linearbroadcasted(-, a, b)
        @test copy(x) ≈ a - b

        # Unary minus
        x = linearbroadcasted(-, a)
        @test copy(x) ≈ -a

        # Division
        x = linearbroadcasted(/, a, 2)
        @test copy(x) ≈ a / 2

        # Left division
        x = linearbroadcasted(\, 2, a)
        @test copy(x) ≈ a / 2

        # Scaling distributes over AddBroadcasted
        ab = linearbroadcasted(+, a, b)
        x = linearbroadcasted(*, 3, ab)
        @test copy(x) ≈ 3a + 3b

        # Conjugation distributes over AddBroadcasted
        x = linearbroadcasted(conj, ab)
        @test copy(x) ≈ conj(a) + conj(b)

        # Conjugation distributes over Mul
        m = TA.Mul(a, b)
        x = linearbroadcasted(conj, m)
        @test copy(x) ≈ conj(a) * conj(b)
    end
    @testset "AddBroadcasted flattening" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)
        c = randn(ComplexF64, 2, 2)

        # AddBroadcasted + array flattens
        ab = linearbroadcasted(+, a, b)
        x = linearbroadcasted(+, ab, c)
        @test TA.addends(x) === (a, b, c)

        # array + AddBroadcasted flattens
        x = linearbroadcasted(+, c, ab)
        @test TA.addends(x) === (c, a, b)

        # AddBroadcasted + AddBroadcasted flattens
        cd = linearbroadcasted(+, c, a)
        x = linearbroadcasted(+, ab, cd)
        @test TA.addends(x) === (a, b, c, a)
    end
end
