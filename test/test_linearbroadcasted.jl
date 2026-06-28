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
        @test x ≡ TA.ConjArray(a)
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
    @testset "tryflattenlinear" begin
        a = randn(ComplexF64, 2, 2)
        b = randn(ComplexF64, 2, 2)

        # Linear expressions convert successfully
        @test TA.tryflattenlinear(BC.broadcasted(*, 2, a)) ≡ linearbroadcasted(*, 2, a)
        @test TA.tryflattenlinear(BC.broadcasted(conj, a)) ≡ linearbroadcasted(conj, a)
        @test TA.tryflattenlinear(BC.broadcasted(+, a, b)) ≡ linearbroadcasted(+, a, b)
        @test TA.tryflattenlinear(BC.broadcasted(identity, a)) ≡ a

        # Nested linear expression
        bc = BC.broadcasted(+, BC.broadcasted(*, 2, a), BC.broadcasted(*, 3, b))
        @test copy(TA.tryflattenlinear(bc)) ≈ 2a + 3b

        # Nonlinear expression returns nothing
        @test TA.tryflattenlinear(BC.broadcasted(exp, a)) === nothing
        @test TA.tryflattenlinear(BC.broadcasted(+, a, BC.broadcasted(exp, b))) === nothing
    end
    @testset "linearbroadcasted algebra" begin
        a = randn(ComplexF64, 3, 3)

        # Scaling absorbs coefficients
        @test linearbroadcasted(*, 3, linearbroadcasted(*, 2, a)) ≡
            TA.ScaledBroadcasted(6, a)

        # Conjugation of scaled
        x = linearbroadcasted(conj, linearbroadcasted(*, 2im, a))
        @test x ≡ TA.ScaledBroadcasted(-2im, TA.ConjArray(a))

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
    @testset "similar(::AddBroadcasted) with LinearBroadcasted addends" begin
        a = randn(ComplexF64, 3, 4)
        b = randn(ComplexF64, 3, 4)

        # Addends are ScaledBroadcasted, not AbstractArray
        lb = linearbroadcasted(+, linearbroadcasted(*, 2, a), linearbroadcasted(*, 3, b))
        s = similar(lb)
        @test size(s) == (3, 4)
        @test eltype(s) === ComplexF64
    end
    @testset "_compose_op" begin
        @test TA._compose_op(identity, identity) === identity
        @test TA._compose_op(identity, conj) === conj
        @test TA._compose_op(conj, identity) === conj
        @test TA._compose_op(conj, conj) === identity
        f = TA._compose_op(sqrt, conj)
        @test f isa ComposedFunction
    end
    @testset "Broadcasted(::LinearBroadcasted) round-trip" begin
        a = randn(ComplexF64, 3, 3)
        b = randn(ComplexF64, 3, 3)

        lb = linearbroadcasted(+, linearbroadcasted(*, 2, a), linearbroadcasted(conj, b))
        bc = BC.Broadcasted(lb)
        @test bc isa BC.Broadcasted
        @test copy(bc) ≈ 2a + conj(b)
    end
    @testset "add! and copyto! with LinearBroadcasted" begin
        a = randn(ComplexF64, 3, 3)
        b = randn(ComplexF64, 3, 3)

        # add! with ScaledBroadcasted
        dest = zeros(ComplexF64, 3, 3)
        TA.add!(dest, linearbroadcasted(*, 2, a), true, false)
        @test dest ≈ 2a

        # add! with AddBroadcasted
        dest = zeros(ComplexF64, 3, 3)
        TA.add!(dest, linearbroadcasted(+, a, b), true, false)
        @test dest ≈ a + b

        # add! with a ConjArray
        dest = zeros(ComplexF64, 3, 3)
        TA.add!(dest, linearbroadcasted(conj, a), true, false)
        @test dest ≈ conj(a)

        # add! with β accumulation
        dest = ones(ComplexF64, 3, 3)
        TA.add!(dest, linearbroadcasted(*, 2, a), 3, 1)
        @test dest ≈ ones(ComplexF64, 3, 3) + 6a
    end
    @testset "0-dimensional permutedimsopadd!" begin
        a = fill(3.0 + 2.0im)
        dest = fill(1.0 + 0.0im)
        TA.permutedimsopadd!(dest, identity, a, (), 2, 3)
        @test dest[] ≈ 3 * 1 + 2 * a[]

        dest = fill(0.0 + 0.0im)
        TA.permutedimsopadd!(dest, conj, a, (), 1, 0)
        @test dest[] ≈ conj(a[])
    end
end
