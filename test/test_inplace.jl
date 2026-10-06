using LinearAlgebra: Diagonal
using TensorAlgebra: twist!
using Test: @test, @testset

@testset "twist! is a no-op on non-graded arrays" begin
    a = randn(2, 2)
    @test twist!(copy(a), (1,)) == a
    @test twist!(copy(a), (1, 2)) == a
    d = Diagonal(randn(3))
    @test twist!(copy(d), (1,)) == d
end
