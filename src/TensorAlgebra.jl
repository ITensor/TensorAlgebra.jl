module TensorAlgebra

export contract, contract!, eigen, eigvals, factorize, gram_eigh_full,
    gram_eigh_full_with_pinv, left_null, left_orth, left_polar, lq, qr,
    right_null, right_orth, right_polar, orth, polar, svd, svdvals

if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public contractopadd!, matricizeop"))
end

include("MatrixAlgebra.jl")
include("blockedtuple.jl")
include("blockedpermutation.jl")
include("BaseExtensions/BaseExtensions.jl")
include("permutedimsadd.jl")
include("matricize.jl")
include("contract/contractalgorithm.jl")
include("contract/contract.jl")
include("contract/contract_labels.jl")
include("contract/blockedperms.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize.jl")
include("factorizations.jl")
include("matrixfunctions.jl")
include("similar_map.jl")
include("projectto.jl")
include("linearbroadcasted.jl")

end
