using MatrixAlgebraKit: MatrixAlgebraKit
using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "Test exports" begin
    exports = [
        :TensorAlgebra,
        :contract,
        :contract!,
        :contractalign,
        :dual,
        :isdual,
        :MatrixAlgebra,
    ]
    # `public` (Julia 1.11+) adds names to `names()`; include them on 1.11+.
    if VERSION >= v"1.11.0-DEV.469"
        append!(
            exports,
            [
                :AbstractAlgorithm, :add!, :AddBroadcasted,
                :addends, :allocate_output, :allocate_project, :arguments, :axes,
                :bipartition, :bipartition_axes, :biperm, :bipermutedims,
                :bipermutedims!, :bipermutedimsopadd!, :cat_axis, :cat_similar,
                :check_input, :concatenate, :concatenate!, :ConjBroadcasted,
                :contractadd!, :ContractAlgorithm, :contractopadd!, :contractperm,
                :contractperm!,
                :contractpermadd!, :contractpermalign, :contractpermopadd!, :data,
                :datatype,
                :default_algorithm, :dims2cat, :directsum, :eig_full, :eig_trunc,
                :eig_vals, :eigh_full, :eigh_trunc, :eigh_vals, :fill_map,
                :flattenlinear, :infer_aux_space, :invsqrth_safe, :is_output_view,
                :is_projected, :isidentitybiperm, :label_type, :left_null, :left_orth,
                :left_polar, :LinearBroadcasted, :linearbroadcasted, :lq_compact,
                :lq_full, :matricize, :MatricizeContract, :matricizeop, :matricizeop!,
                :matricizeopcopy, :matricizeopview, :MatricizeStyle, :MATRIX_FUNCTIONS,
                :ndims, :ndims_codomain, :ndims_domain, :one, :ones_map, :operation,
                :output_axes,
                :PermutedDims, :permuteddims, :permutedims, :permutedims!,
                :permutedimsadd!, :permutedimsop, :permutedimsopadd!, :project,
                :project!, :project_aux, :project_hermitian, :projectto!, :qr_compact,
                :qr_full, :rand_map, :randn_map, :right_null, :right_orth, :right_polar,
                :scalar, :scale!, :ScaledBroadcasted, :select_algorithm, :similar_map,
                :size, :sqrth_invsqrth_safe, :sqrth_safe, :sum, :svd_compact, :svd_full,
                :svd_trunc, :svd_vals, :TensorOperationsContract, :to_range, :tr,
                :trivialrange, :tryflattenlinear, :tryproject, :tryproject_aux,
                :unchecked_project, :unchecked_project_aux, :ungrade, :unmatricize,
                :unmatricize!, :unmatricize_factors, :unproject, :unscaled, :zero!,
                :zeros_map,
            ]
        )
    end
    @test issetequal(names(TensorAlgebra), exports)

    # The matrix-level factorizations are `public`, not exported: the names MatrixAlgebraKit
    # also exports would otherwise collide in a session loading both packages, and the
    # `MatrixAlgebra` spellings are reached through the submodule, which is exported.
    exported = filter(n -> Base.isexported(TensorAlgebra, n), names(TensorAlgebra))
    @test issetequal(
        exported,
        [
            :TensorAlgebra, :contract, :contract!, :contractalign, :dual, :isdual,
            :MatrixAlgebra,
        ]
    )
    @test isempty(
        intersect(
            exported,
            filter(n -> Base.isexported(MatrixAlgebraKit, n), names(MatrixAlgebraKit))
        )
    )

    exports = [
        :MatrixAlgebra,
        :invsqrt_diag_safe,
        :invsqrth_safe,
        :var"one!",
        :pow_diag_safe,
        :pow_diag_safe!,
        :powh_safe,
        :sqrt_diag_safe,
        :sqrth_invsqrth_safe,
        :sqrth_safe,
    ]
    @test issetequal(names(TensorAlgebra.MatrixAlgebra), exports)
end
