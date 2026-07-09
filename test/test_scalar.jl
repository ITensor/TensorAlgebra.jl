using TensorAlgebra: scalar
using Test: @test, @testset

@testset "scalar" begin
    @test scalar(fill(3.0)) === 3.0
    a = Array{Float64, 0}(undef)
    a[] = 5.0
    @test scalar(a) === 5.0
    @test scalar(fill(2 + 3im)) === 2 + 3im
end
