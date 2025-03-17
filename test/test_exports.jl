using TensorAlgebra: TensorAlgebra
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :TensorAlgebra,
    :contract,
    :contract!,
    :eigen,
    :eigvals,
    :left_null,
    :lq,
    :qr,
    :right_null,
    :svd,
    :svdvals,
  ]
  @test issetequal(names(TensorAlgebra), exports)
end
