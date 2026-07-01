module TensorAlgebra

export contract, contract!, eig_full, eig_trunc, eig_vals, eigh_full, eigh_trunc,
    eigh_vals, gram_eigh_full, gram_eigh_full_with_pinv, left_null, left_orth,
    left_polar, lq_compact, lq_full, qr_compact, qr_full, right_null, right_orth,
    right_polar, svd_compact, svd_full, svd_trunc, svd_vals

if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public biperm, bipartition, contractopadd!, label_type, matricizeopperm, to_range, zero!, scale!, permuteddims, conjed, ConjArray"
        )
    )
end

include("inplace.jl")
include("MatrixAlgebra.jl")
include("bituple.jl")
include("permutedimsadd.jl")
include("conjarray.jl")
include("matricize.jl")
include("diagonal.jl")
include("to_range.jl")
include("contract/contractalgorithm.jl")
include("contract/contract.jl")
include("contract/contract_labels.jl")
include("contract/biperms.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize.jl")
include("factorizations.jl")
include("matrixfunctions.jl")
include("similar_map.jl")
include("projectto.jl")
include("linearbroadcasted.jl")

end
