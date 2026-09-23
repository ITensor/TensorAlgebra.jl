module TensorAlgebra

export contract, contract!, contractalign, dual, isdual, MatrixAlgebra

if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public AbstractAlgorithm, add!, AddBroadcasted, addends, allocate_output, allocate_project, arguments, axes, bipartition, bipartition_axes, biperm, bipermutedims, bipermutedims!, bipermutedimsopadd!, cat_axis, cat_similar, check_input, concatenate, concatenate!, ConjBroadcasted, contractadd!, ContractAlgorithm, contractopadd!, contractperm, contractperm!, contractpermadd!, contractpermalign, contractpermopadd!, data, datatype, default_algorithm, dims2cat, directsum, eig_full, eig_trunc, eig_vals, eigh_full, eigh_trunc, eigh_vals, fill_map, flattenlinear, infer_aux_space, invsqrth_safe, is_output_view, is_projected, isidentitybiperm, label_type, left_null, left_orth, left_polar, LinearBroadcasted, linearbroadcasted, lq_compact, lq_full, matricize, MatricizeContract, matricizeop, matricizeop!, matricizeopcopy, matricizeopview, MatricizeStyle, MATRIX_FUNCTIONS, ndims, ndims_codomain, one, ones_map, operation, output_axes, PermutedDims, permuteddims, permutedims, permutedims!, permutedimsadd!, permutedimsop, permutedimsopadd!, project, project!, project_aux, project_hermitian, projectto!, qr_compact, qr_full, rand_map, randn_map, right_null, right_orth, right_polar, scalar, scale!, ScaledBroadcasted, select_algorithm, similar_map, size, sqrth_invsqrth_safe, sqrth_safe, sum, svd_compact, svd_full, svd_trunc, svd_vals, TensorOperationsContract, to_range, tr, trivialrange, tryflattenlinear, tryproject, tryproject_aux, unchecked_project, unchecked_project_aux, ungrade, unmatricize, unmatricize!, unmatricize_factors, unproject, unscaled, zero!, zeros_map"
        )
    )
end

include("interface.jl")
include("datatype.jl")
include("inplace.jl")
include("MatrixAlgebra.jl")
include("bituple.jl")
include("permutedimsadd.jl")
include("matricize.jl")
include("concatenate.jl")
include("directsum.jl")
include("dual.jl")
include("to_range.jl")
include("algorithm.jl")
include("contract/contractalgorithm.jl")
include("contract/contract.jl")
include("contract/contract_labels.jl")
include("contract/biperms.jl")
include("contract/allocate_output.jl")
include("diagonal.jl")
include("contract/contract_matricize.jl")
include("factorizations.jl")
include("matrixfunctions.jl")
include("similar_map.jl")
include("projectto.jl")
include("linearbroadcasted.jl")

end
