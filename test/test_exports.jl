using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "Test exports" begin
    exports = [
        :TensorAlgebra,
        :contract,
        :contract!,
        :eigen,
        :eigvals,
        :factorize,
        :left_null,
        :left_orth,
        :left_polar,
        :lq,
        :orth,
        :polar,
        :qr,
        :right_null,
        :right_orth,
        :right_polar,
        :svd,
        :svdvals,
    ]
    @test issetequal(names(TensorAlgebra), exports)

    exports = [
        :MatrixAlgebra,
        :eigen,
        :eigen!!,
        :eigvals,
        :eigvals!!,
        :factorize,
        :factorize!!,
        :lq,
        :lq!!,
        :orth,
        :orth!!,
        :polar,
        :polar!!,
        :qr,
        :qr!!,
        :svd,
        :svd!!,
        :svdvals,
        :svdvals!!,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
