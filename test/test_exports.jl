using TensorAlgebra: TensorAlgebra
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :TensorAlgebra, :contract, :contract!, :eig, :eigvals, :lq, :qr, :svd, :svdvals
  ]
  @test issetequal(names(TensorAlgebra), exports)
end
