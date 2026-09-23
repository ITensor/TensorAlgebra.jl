module TensorAlgebraTensorOperationsExt

using TensorAlgebra: TensorAlgebra as TA, TensorOperationsContract
using TensorOperations: TensorOperations as TO

# `TensorOperationsContract` stores `nothing` to mean "TensorOperations' default"; resolve
# those here, where the defaults can be named.
function backend(algorithm::TensorOperationsContract)
    return @something algorithm.backend TO.DefaultBackend()
end
function allocator(algorithm::TensorOperationsContract)
    return @something algorithm.allocator TO.DefaultAllocator()
end

# Construct via the `AbstractContractAlgorithm` public constructor seam as well.
function TA.AbstractContractAlgorithm(backend::TO.AbstractBackend)
    return TensorOperationsContract(; backend)
end
function TA.AbstractContractAlgorithm(backend::TO.AbstractBackend, allocator)
    return TensorOperationsContract(; backend, allocator)
end

# Using TensorOperations backends as TensorAlgebra implementations
# ----------------------------------------------------------------

function TA.contractpermopadd!(
        algorithm::TensorOperationsContract,
        a_dest, perm_dest_codomain, perm_dest_domain,
        op1, a1, perm1_codomain, perm1_domain,
        op2, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    permblocks1 = Tuple.((perm1_codomain, perm1_domain))
    permblocks2 = Tuple.((perm2_codomain, perm2_domain))
    permblocks_dest = Tuple.((perm_dest_codomain, perm_dest_domain))
    conj1 = op1 === conj
    conj2 = op2 === conj
    a1′ = (op1 === identity || op1 === conj) ? a1 : op1.(a1)
    a2′ = (op2 === identity || op2 === conj) ? a2 : op2.(a2)
    return TO.tensorcontract!(
        a_dest, a1′, permblocks1, conj1, a2′, permblocks2, conj2,
        permblocks_dest, α, β, backend(algorithm), allocator(algorithm)
    )
end

# Using TensorAlgebra implementations as TensorOperations backends
# ----------------------------------------------------------------
function TO.tensorcontract!(
        a_dest::AbstractArray,
        a1::AbstractArray, permblocks1::TO.Index2Tuple, conj1::Bool,
        a2::AbstractArray, permblocks2::TO.Index2Tuple, conj2::Bool,
        permblocks_dest::TO.Index2Tuple,
        α::Number, β::Number,
        backend::TA.AbstractContractAlgorithm,
        allocator
    )
    op1 = conj1 ? conj : identity
    op2 = conj2 ? conj : identity
    return TA.contractpermopadd!(
        backend,
        a_dest, permblocks_dest...,
        op1, a1, permblocks1...,
        op2, a2, permblocks2...,
        α, β
    )
end

# For now no trace/add is supported, so simply reselect default backend from TensorOperations
function TO.tensortrace!(
        a_dest::AbstractArray,
        a_src::AbstractArray,
        permblocks_src::TO.Index2Tuple,
        permblocks_dest::TO.Index2Tuple,
        conj_src::Bool,
        α::Number, β::Number,
        ::TA.AbstractContractAlgorithm,
        allocator
    )
    return TO.tensortrace!(
        a_dest, a_src, permblocks_src,
        permblocks_dest, conj_src,
        α, β, TO.DefaultBackend(), allocator
    )
end
function TO.tensoradd!(
        a_dest::AbstractArray,
        a_src::AbstractArray,
        permblocks_src::TO.Index2Tuple,
        conj_src::Bool,
        α::Number, β::Number,
        ::TA.AbstractContractAlgorithm,
        allocator
    )
    return TO.tensoradd!(
        a_dest, a_src, permblocks_src, conj_src, α, β, TO.DefaultBackend(), allocator
    )
end

end
