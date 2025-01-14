using TensorAlgebra: TensorAlgebra
using Test: @test, @testset
@testset "Test exports" begin
  exports = [:TensorAlgebra, :contract, :contract!]
  @test issetequal(names(TensorAlgebra), exports)
end
