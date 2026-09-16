using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "Test exports" begin
    exports = [
        :TensorAlgebra,
        :contract,
        :contract!,
        :contractalign,
        :dual,
        :eig_full,
        :eig_trunc,
        :eig_vals,
        :eigh_full,
        :eigh_trunc,
        :eigh_vals,
        :invsqrth_safe,
        :isdual,
        :left_null,
        :left_orth,
        :left_polar,
        :lq_compact,
        :lq_full,
        :project_hermitian,
        :qr_compact,
        :qr_full,
        :right_null,
        :right_orth,
        :right_polar,
        :sqrth_safe,
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
                :allocate_output, :biperm, :bipartition, :cat_similar, :check_input,
                :concatenate, :concatenate!, :ContractAlgorithm, :contractopadd!,
                :contractperm, :contractperm!, :contractpermadd!, :contractpermopadd!,
                :data,
                :datatype, :directsum,
                :flattenlinear, :is_output_view, :label_type,
                :matricize, :matricizeop, :matricizeop!, :matricizeopcopy, :matricizeopview,
                :default_algorithm, :output_axes, :permutedims, :select_algorithm,
                :permutedims!, :scalar, :similar_map,
                :TensorOperationsAlgorithm,
                :to_range, :tr, :tryflattenlinear, :ungrade, :zero!, :scale!,
                :permuteddims, :PermutedDims,
            ]
        )
    end
    @test issetequal(names(TensorAlgebra), exports)

    exports = [
        :MatrixAlgebra,
        :invsqrt_diag_safe,
        :invsqrth_safe,
        :pow_diag_safe,
        :pow_diag_safe!,
        :powh_safe,
        :sqrt_diag_safe,
        :sqrth_safe,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
