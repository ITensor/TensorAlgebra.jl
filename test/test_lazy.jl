using TensorAlgebra: TensorAlgebra as TA, +ₗ, *ₗ, conjed
using Test: @test, @testset

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
