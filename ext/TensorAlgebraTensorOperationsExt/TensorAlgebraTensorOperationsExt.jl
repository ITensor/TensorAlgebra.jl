module TensorAlgebraTensorOperationsExt

using TensorAlgebra: TensorAlgebra as TA
using TensorOperations: TensorOperations as TO

"""
    TensorOperationsAlgorithm(backend::AbstractBackend)

Wrapper type for making a TensorOperations backend work as a TensorAlgebra algorithm.
"""
struct TensorOperationsAlgorithm{B <: TO.AbstractBackend} <: TA.ContractAlgorithm
    backend::B
end

TA.ContractAlgorithm(backend::TO.AbstractBackend) = TensorOperationsAlgorithm(backend)

# Using TensorOperations backends as TensorAlgebra implementations
# ----------------------------------------------------------------

# not in-place
function TA.contract(
        algorithm::TensorOperationsAlgorithm,
        perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain
    )
    permblocks1 = Tuple.((perm1_codomain, perm1_domain))
    permblocks2 = Tuple.((perm2_codomain, perm2_domain))
    permblocks_dest = Tuple.((perm_dest_codomain, perm_dest_domain))
    conj1, conj2 = false, false
    α = true
    return TO.tensorcontract(
        a1, permblocks1, conj1, a2, permblocks2, conj2,
        permblocks_dest, α, algorithm.backend
    )
end

function TA.contract(
        algorithm::TensorOperationsAlgorithm,
        labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2
    )
    permblocks1, permblocks2, permblocks_dest =
        TO.contract_indices(labels1, labels2, labels_dest)
    conj1, conj2 = false, false
    α = true
    return TO.tensorcontract(
        a1, permblocks1, conj1, a2, permblocks2, conj2,
        permblocks_dest, α, algorithm.backend
    )
end

# in-place
function TA.contractadd!(
        algorithm::TensorOperationsAlgorithm,
        a_dest::AbstractArray, perm_dest_codomain, perm_dest_domain,
        a1::AbstractArray, perm1_codomain, perm1_domain,
        a2::AbstractArray, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    permblocks1 = Tuple.((perm1_codomain, perm1_domain))
    permblocks2 = Tuple.((perm2_codomain, perm2_domain))
    permblocks_dest = Tuple.((perm_dest_codomain, perm_dest_domain))
    conj1, conj2 = false, false
    return TO.tensorcontract!(
        a_dest, a1, permblocks1, conj1, a2, permblocks2, conj2,
        permblocks_dest, α, β, algorithm.backend
    )
end

function TA.contractadd!(
        algorithm::TensorOperationsAlgorithm,
        a_dest::AbstractArray, labels_dest,
        a1::AbstractArray, labels1,
        a2::AbstractArray, labels2,
        α::Number, β::Number
    )
    permblocks1, permblocks2, permblocks_dest =
        TO.contract_indices(labels1, labels2, labels_dest)
    conj1, conj2 = false, false
    return TO.tensorcontract!(
        a_dest, a1, permblocks1, conj1, a2, permblocks2, conj2,
        permblocks_dest, α, β, algorithm.backend
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
        backend::TA.ContractAlgorithm,
        allocator
    )
    # TODO: FIXME: Use `conjed` to do the conjugation lazily.
    a1′ = conj1 ? conj(a1) : a1
    a2′ = conj2 ? conj(a2) : a2
    return TA.contractadd!(
        backend,
        a_dest, permblocks_dest...,
        a1′, permblocks1...,
        a2′, permblocks2...,
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
        ::TA.ContractAlgorithm,
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
        ::TA.ContractAlgorithm,
        allocator
    )
    return TO.tensoradd!(
        a_dest, a_src, permblocks_src, conj_src, α, β, TO.DefaultBackend(), allocator
    )
end

end
