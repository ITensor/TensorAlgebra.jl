module TensorAlgebra

export contract, contract!, contractalign, dual, eig_full, eig_trunc, eig_vals, eigh_full,
    eigh_trunc,
    eigh_vals, invsqrth_safe, isdual, left_null,
    left_orth, left_polar, lq_compact, lq_full, project_hermitian, qr_compact,
    qr_full, right_null, right_orth, right_polar, sqrth_safe,
    svd_compact, svd_full, svd_trunc, svd_vals

if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public allocate_output, biperm, bipartition, cat_similar, check_input, concatenate, concatenate!, ContractAlgorithm, contractadd!, contractopadd!, contractperm, contractperm!, contractpermadd!, contractpermopadd!, data, datatype, directsum, flattenlinear, is_output_view, label_type, matricize, matricizeop, matricizeop!, matricizeopcopy, matricizeopview, output_axes, select_algorithm, default_algorithm, permutedims, permutedims!, scalar, similar_map, TensorOperationsAlgorithm, to_range, tr, tryflattenlinear, ungrade, zero!, scale!, permuteddims, PermutedDims"
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
