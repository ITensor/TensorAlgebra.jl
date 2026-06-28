using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "to_range" begin
    @testset "Integer -> Base.OneTo" begin
        r = TensorAlgebra.to_range(3)
        @test r === Base.OneTo(3)
    end

    @testset "range passthrough is idempotent" begin
        for r in (Base.OneTo(4), 2:5)
            @test TensorAlgebra.to_range(r) === r
        end
    end
end
