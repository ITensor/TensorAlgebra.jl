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
        :eigvals,
        :factorize,
        :gram_eigh_full,
        :gram_eigh_full_with_pinv,
        :invsqrt_diag_safe,
        :invsqrth_safe,
        :lq,
        :orth,
        :polar,
        :pow_diag_safe,
        :powh_safe,
        :qr,
        :sqrt_diag_safe,
        :sqrth_safe,
        :svd,
        :svdvals,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
