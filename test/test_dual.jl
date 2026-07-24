using TensorAlgebra: TensorAlgebra, dual, isdual
using Test: @test, @test_throws, @testset

@testset "dual/isdual fallbacks on ranges" begin
    # An ordinary range has no arrow: never dual, and its own dual.
    for r in (Base.OneTo(4), 2:5)
        @test isdual(r) == false
        @test dual(r) === r
    end
    # No universal fallback: a type with no duality concept errors instead of
    # silently returning a default.
    @test_throws MethodError isdual(3)
    @test_throws MethodError dual(3)
end
