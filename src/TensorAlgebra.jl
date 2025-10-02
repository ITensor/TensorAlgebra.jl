module TensorAlgebra

export contract,
    contract!,
    eigen,
    eigvals,
    factorize,
    left_null,
    left_orth,
    left_polar,
    lq,
    qr,
    right_null,
    right_orth,
    right_polar,
    orth,
    polar,
    svd,
    svdvals

include("MatrixAlgebra.jl")
include("blockedtuple.jl")
include("blockedpermutation.jl")
include("BaseExtensions/BaseExtensions.jl")
include("matricize.jl")
include("contract/contract.jl")
include("contract/output_labels.jl")
include("contract/blockedperms.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize/contract.jl")
include("factorizations.jl")
include("matrixfunctions.jl")

end
