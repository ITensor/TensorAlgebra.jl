import FunctionImplementations as FI
using Base.Broadcast: Broadcast as BC
using TensorAlgebra: TensorAlgebra as TA, *ₗ, +ₗ, /ₗ, conjed
using Test: @test, @test_broken, @test_throws, @testset

@testset "lazy arrays" begin
    @testset "lazy array operations" begin
        a = randn(ComplexF64, 3, 3)
        b = randn(ComplexF64, 3, 3)
        c = randn(ComplexF64, 3, 3)

        x = 2 *ₗ a
        @test x ≡ TA.ScaledArray(2, a)
        @test copy(x) ≈ 2a

        x = conjed(a)
        @test x ≡ TA.ConjArray(a)
        @test copy(x) ≈ conj(a)
        @test conj(x) ≈ a

        x = a +ₗ b
        @test x ≡ TA.AddArray(a, b)
        @test copy(x) ≈ a + b

        x = a *ₗ b
        @test x ≡ TA.MulArray(a, b)
        @test copy(x) ≈ a * b

        x = a *ₗ b +ₗ c
        @test x ≡ TA.AddArray(TA.MulArray(a, b), c)
        @test copy(x) ≈ a *ₗ b .+ c ≈ a * b + c

        x = 2 *ₗ a *ₗ b +ₗ 3 *ₗ c
        @test x ≡ TA.AddArray(TA.ScaledArray(2, TA.MulArray(a, b)), TA.ScaledArray(3, c))
        @test copy(x) ≈ 2 .* a *ₗ b .+ 3 .* c ≈ 2 * a * b + 3 * c
    end
    @testset "adjoint" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        x = (2 *ₗ a)'
        @test x ≡ 2 *ₗ a'
        @test copy(x) ≈ 2a'

        x = conjed(a)'
        @test x ≡ transpose(a)
        @test copy(x) ≈ permutedims(a)

        x = (a +ₗ b)'
        @test x ≡ a' +ₗ b'
        @test copy(x) ≈ a' + b'

        x = (a *ₗ b)'
        @test x ≡ b' *ₗ a'
        @test copy(x) ≈ b' * a'
    end
    @testset "transpose" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        x = transpose(2 *ₗ a)
        @test x ≡ 2 *ₗ transpose(a)
        @test copy(x) ≈ 2transpose(a)

        x = transpose(conjed(a))
        @test x ≡ adjoint(a)
        @test copy(x) ≈ permutedims(conj(a))

        x = transpose(a +ₗ b)
        @test x ≡ transpose(a) +ₗ transpose(b)
        @test copy(x) ≈ transpose(a) + transpose(b)

        x = transpose(a *ₗ b)
        @test x ≡ transpose(b) *ₗ transpose(a)
        @test copy(x) ≈ transpose(b) * transpose(a)
    end
    @testset "permuteddims" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)
        perm = (2, 1)

        x = FI.permuteddims(2 *ₗ a, perm)
        @test x ≡ 2 *ₗ FI.permuteddims(a, perm)
        @test copy(x) ≈ 2permutedims(a, perm)

        x = FI.permuteddims(conjed(a), perm)
        @test x ≡ conjed(FI.permuteddims(a, perm))
        @test copy(x) ≈ conj(permutedims(a, perm))

        x = FI.permuteddims(a +ₗ b, perm)
        @test x ≡ FI.permuteddims(a, perm) +ₗ FI.permuteddims(b, perm)
        @test copy(x) ≈ permutedims(a, perm) + permutedims(b, perm)

        x = FI.permuteddims(a *ₗ b, perm)
        @test x ≡ PermutedDimsArray(a *ₗ b, perm)
        @test copy(x) ≈ permutedims(a * b, perm)
    end
    @testset "linear broadcast lowering" begin
        a = randn(ComplexF64, 2, 2)
        style = BC.DefaultArrayStyle{2}()

        @test TA.broadcasted_linear(identity, a) ≡ a
        @test TA.broadcasted_linear(Base.Fix1(*, 2), a) ≡ 2 *ₗ a
        @test TA.broadcasted_linear(Base.Fix2(*, 2), a) ≡ a *ₗ 2
        @test TA.broadcasted_linear(Base.Fix2(/, 2), a) ≡ a /ₗ 2
        @test TA.broadcasted_linear(style, identity, a) ≡ a
        @test TA.broadcasted_linear(style, Base.Fix1(*, 2), a) ≡ 2 *ₗ a
        @test TA.broadcasted_linear(style, Base.Fix2(*, 2), a) ≡ a *ₗ 2
        @test TA.broadcasted_linear(style, Base.Fix2(/, 2), a) ≡ a /ₗ 2
        @test TA.broadcasted_linear(style, conj, a) ≡ conjed(a)
        @test_throws ArgumentError TA.broadcasted_linear(style, exp, a)
    end
    @testset "scalar getindex" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        @test (2 *ₗ a)[1, 2] == 2 * a[1, 2]
        @test conjed(a)[2, 1] == conj(a[2, 1])
        @test (a +ₗ b)[2, 2] == a[2, 2] + b[2, 2]
        @test (a *ₗ b)[1, 2] ≈ (a * b)[1, 2]
        @test (a *ₗ b)[3] ≈ (a * b)[3]
    end
end
