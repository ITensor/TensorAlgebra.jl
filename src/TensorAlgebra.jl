module TensorAlgebra

export contract, contract!, eigen, eigvals, lq, left_null, qr, right_null, svd, svdvals

include("blockedtuple.jl")
include("blockedpermutation.jl")
include("BaseExtensions/BaseExtensions.jl")
include("fusedims.jl")
include("splitdims.jl")
include("contract/contract.jl")
include("contract/output_labels.jl")
include("contract/blockedperms.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize/contract.jl")
include("factorizations.jl")

end
