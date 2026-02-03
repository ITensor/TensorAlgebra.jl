module TensorAlgebra

export contract, contract!, eigen, eigvals, factorize, left_null, left_orth, left_polar,
    lq, qr, right_null, right_orth, right_polar, orth, polar, svd, svdvals

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
include("lazyarrays.jl")

end
