import FunctionImplementations as FI
using Base.Broadcast: Broadcast as BC
using TensorAlgebra: TensorAlgebra as TA, LBF
using Test: @test, @test_throws, @testset

@testset "LinearBroadcasted and Mul" begin
    @testset "construction and materialization" begin
        a = randn(ComplexF64, 3, 3)
        b = randn(ComplexF64, 3, 3)
        c = randn(ComplexF64, 3, 3)

        x = LBF(*)(2, a)
        @test x ≡ TA.ScaledBroadcasted(2, a)
        @test copy(x) ≈ 2a

        x = LBF(conj)(a)
        @test x ≡ TA.ConjBroadcasted(a)
        @test copy(x) ≈ conj(a)
        @test conj(x) ≈ a

        x = LBF(+)(a, b)
        @test x ≡ TA.AddBroadcasted(a, b)
        @test copy(x) ≈ a + b

        x = TA.Mul(a, b)
        @test copy(x) ≈ a * b

        x = LBF(+)(TA.Mul(a, b), c)
        @test x ≡ TA.AddBroadcasted(TA.Mul(a, b), c)
        @test copy(x) ≈ a * b + c

        x = LBF(+)(LBF(*)(2, TA.Mul(a, b)), LBF(*)(3, c))
        @test x ≡ TA.AddBroadcasted(
            TA.ScaledBroadcasted(2, TA.Mul(a, b)), TA.ScaledBroadcasted(3, c)
        )
        @test copy(x) ≈ 2 * a * b + 3 * c
    end
    @testset "adjoint" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        x = LBF(*)(2, a)'
        @test x ≡ LBF(*)(2, a')
        @test copy(x) ≈ 2a'

        x = LBF(conj)(a)'
        @test x ≡ transpose(a)
        @test copy(x) ≈ permutedims(a)

        x = LBF(+)(a, b)'
        @test x ≡ LBF(+)(a', b')
        @test copy(x) ≈ a' + b'

        x = TA.Mul(a, b)'
        @test x ≡ TA.Mul(b', a')
        @test copy(x) ≈ b' * a'
    end
    @testset "transpose" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        x = transpose(LBF(*)(2, a))
        @test x ≡ LBF(*)(2, transpose(a))
        @test copy(x) ≈ 2transpose(a)

        x = transpose(LBF(conj)(a))
        @test x ≡ adjoint(a)
        @test copy(x) ≈ permutedims(conj(a))

        x = transpose(LBF(+)(a, b))
        @test x ≡ LBF(+)(transpose(a), transpose(b))
        @test copy(x) ≈ transpose(a) + transpose(b)

        x = transpose(TA.Mul(a, b))
        @test x ≡ TA.Mul(transpose(b), transpose(a))
        @test copy(x) ≈ transpose(b) * transpose(a)
    end
    @testset "permuteddims" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)
        perm = (2, 1)

        x = FI.permuteddims(LBF(*)(2, a), perm)
        @test x ≡ LBF(*)(2, FI.permuteddims(a, perm))
        @test copy(x) ≈ 2permutedims(a, perm)

        x = FI.permuteddims(LBF(conj)(a), perm)
        @test x ≡ LBF(conj)(FI.permuteddims(a, perm))
        @test copy(x) ≈ conj(permutedims(a, perm))

        x = FI.permuteddims(LBF(+)(a, b), perm)
        @test x ≡ LBF(+)(FI.permuteddims(a, perm), FI.permuteddims(b, perm))
        @test copy(x) ≈ permutedims(a, perm) + permutedims(b, perm)

        x = FI.permuteddims(TA.Mul(a, b), perm)
        @test copy(x) ≈ permutedims(a * b, perm)
    end
    @testset "linear broadcast lowering" begin
        a = randn(ComplexF64, 2, 2)
        style = BC.DefaultArrayStyle{2}()

        @test TA.broadcasted_linear(identity, a) ≡ a
        @test TA.broadcasted_linear(Base.Fix1(*, 2), a) ≡ LBF(*)(2, a)
        @test TA.broadcasted_linear(Base.Fix2(*, 2), a) ≡ LBF(*)(a, 2)
        @test TA.broadcasted_linear(Base.Fix2(/, 2), a) ≡ LBF(/)(a, 2)
        @test TA.broadcasted_linear(style, identity, a) ≡ a
        @test TA.broadcasted_linear(style, Base.Fix1(*, 2), a) ≡ LBF(*)(2, a)
        @test TA.broadcasted_linear(style, Base.Fix2(*, 2), a) ≡ LBF(*)(a, 2)
        @test TA.broadcasted_linear(style, Base.Fix2(/, 2), a) ≡ LBF(/)(a, 2)
        @test TA.broadcasted_linear(style, conj, a) ≡ LBF(conj)(a)
        @test_throws ArgumentError TA.broadcasted_linear(style, exp, a)
    end
    @testset "LinearBroadcastFunction algebra" begin
        a = randn(ComplexF64, 3, 3)

        # Scaling absorbs coefficients
        @test LBF(*)(3, LBF(*)(2, a)) ≡ TA.ScaledBroadcasted(6, a)

        # Conjugation of scaled
        x = LBF(conj)(LBF(*)(2im, a))
        @test x ≡ TA.ScaledBroadcasted(-2im, TA.ConjBroadcasted(a))

        # Double conjugation cancels
        @test LBF(conj)(LBF(conj)(a)) ≡ a

        # Subtraction
        b = randn(ComplexF64, 3, 3)
        x = LBF(-)(a, b)
        @test copy(x) ≈ a - b

        # Unary minus
        x = LBF(-)(a)
        @test copy(x) ≈ -a

        # Division
        x = LBF(/)(a, 2)
        @test copy(x) ≈ a / 2

        # Left division
        x = LBF(\)(2, a)
        @test copy(x) ≈ a / 2

        # Scaling distributes over AddBroadcasted
        ab = LBF(+)(a, b)
        x = LBF(*)(3, ab)
        @test copy(x) ≈ 3a + 3b

        # Conjugation distributes over AddBroadcasted
        x = LBF(conj)(ab)
        @test copy(x) ≈ conj(a) + conj(b)

        # Conjugation distributes over Mul
        m = TA.Mul(a, b)
        x = LBF(conj)(m)
        @test copy(x) ≈ conj(a) * conj(b)
    end
    @testset "AddBroadcasted flattening" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)
        c = randn(ComplexF64, 2, 2)

        # AddBroadcasted + array flattens
        ab = LBF(+)(a, b)
        x = LBF(+)(ab, c)
        @test TA.addends(x) === (a, b, c)

        # array + AddBroadcasted flattens
        x = LBF(+)(c, ab)
        @test TA.addends(x) === (c, a, b)

        # AddBroadcasted + AddBroadcasted flattens
        cd = LBF(+)(c, a)
        x = LBF(+)(ab, cd)
        @test TA.addends(x) === (a, b, c, a)
    end
end
