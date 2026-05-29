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
        :gram_eigh_full,
        :gram_eigh_full_with_pinv,
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
    # `public` (Julia 1.11+) adds names to `names()`; include them on 1.11+.
    if VERSION >= v"1.11.0-DEV.469"
        append!(exports, [:contractopadd!, :matricizeop])
    end
    @test issetequal(names(TensorAlgebra), exports)

    exports = [
        :MatrixAlgebra,
        :eigen,
        :eigen!!,
        :eigvals,
        :eigvals!!,
        :factorize,
        :factorize!!,
        :gram_eigh_full,
        :gram_eigh_full!!,
        :gram_eigh_full_with_pinv,
        :gram_eigh_full_with_pinv!!,
        :invsqrt_safe,
        :lq,
        :lq!!,
        :orth,
        :orth!!,
        :polar,
        :polar!!,
        :pow_safe,
        :qr,
        :qr!!,
        :sqrt_safe,
        :svd,
        :svd!!,
        :svdvals,
        :svdvals!!,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
