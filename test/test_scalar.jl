using JLArrays: JLArray
using TensorAlgebra: scalar
using Test: @test, @test_throws, @testset

@testset "scalar" begin
    @test scalar(fill(3.0)) === 3.0
    a = Array{Float64, 0}(undef)
    a[] = 5.0
    @test scalar(a) === 5.0
    @test scalar(fill(2 + 3im)) === 2 + 3im
    @test scalar(JLArray(fill(3.0))) === 3.0
    @test scalar(reshape(JLArray([4.0]), ())) === 4.0
    @test_throws ErrorException scalar(JLArray([3.0]))
end
