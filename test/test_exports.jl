using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "Test exports" begin
    exports = [
        :TensorAlgebra,
        :contract,
        :contract!,
        :eig_full,
        :eig_trunc,
        :eig_vals,
        :eigh_full,
        :eigh_trunc,
        :eigh_vals,
        :gram_eigh_full,
        :gram_eigh_full_with_pinv,
        :left_null,
        :left_orth,
        :left_polar,
        :lq_compact,
        :lq_full,
        :qr_compact,
        :qr_full,
        :right_null,
        :right_orth,
        :right_polar,
        :svd_compact,
        :svd_full,
        :svd_trunc,
        :svd_vals,
    ]
    # `public` (Julia 1.11+) adds names to `names()`; include them on 1.11+.
    if VERSION >= v"1.11.0-DEV.469"
        append!(
            exports,
            [
                :biperm, :bipartition, :contractopadd!, :label_type, :matricizeopperm,
                :to_range, :zero!, :scale!, :permuteddims, :conjed, :ConjArray,
            ]
        )
    end
    @test issetequal(names(TensorAlgebra), exports)

    exports = [
        :MatrixAlgebra,
        :gram_eigh_full,
        :gram_eigh_full_with_pinv,
        :invsqrt_diag_safe,
        :invsqrth_safe,
        :pow_diag_safe,
        :powh_safe,
        :sqrt_diag_safe,
        :sqrth_safe,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
