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

function TA.ContractAlgorithm(backend::TO.AbstractBackend)
    return TensorOperationsContract(; backend)
end
function TA.ContractAlgorithm(backend::TO.AbstractBackend, allocator)
    return TensorOperationsContract(; backend, allocator)
end

# TensorOperations' buffer allocator provides the storage of dense temporaries. Its buffer holds
# raw bytes, so only `Array`s of bits types are served from it.
function TA.similar_map(
        allocator::TO.BufferAllocator, prototype::Array, ::Type{T},
        axes_codomain::Tuple{Vararg{Base.OneTo}}, axes_domain::Tuple{Vararg{Base.OneTo}}
    ) where {T}
    isbitstype(T) || return TA.similar_map(prototype, T, axes_codomain, axes_domain)
    dims = (map(length, axes_codomain)..., map(length, axes_domain)...)
    return TO.tensoralloc(Array{T, length(dims)}, dims, Val(true), allocator)
end
function TA.allocator_checkpoint!(allocator::TO.BufferAllocator)
    return TO.allocator_checkpoint!(allocator)
end
function TA.allocator_reset!(allocator::TO.BufferAllocator, checkpoint)
    return TO.allocator_reset!(allocator, checkpoint)
end

# Using TensorOperations backends as TensorAlgebra implementations
# ----------------------------------------------------------------

function TA.contractpermconjadd!(
        algorithm::TensorOperationsContract,
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    TA.check_input(
        TA.contractpermconjadd!,
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1, a1, perm1_codomain, perm1_domain,
        isconj2, a2, perm2_codomain, perm2_domain
    )
    permblocks1 = Tuple.((perm1_codomain, perm1_domain))
    permblocks2 = Tuple.((perm2_codomain, perm2_domain))
    permblocks_dest = Tuple.((perm_dest_codomain, perm_dest_domain))
    return TO.tensorcontract!(
        a_dest, a1, permblocks1, isconj1, a2, permblocks2, isconj2,
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
        backend::TA.ContractAlgorithm,
        allocator
    )
    return TA.contractpermconjadd!(
        backend,
        a_dest, permblocks_dest...,
        conj1, a1, permblocks1...,
        conj2, a2, permblocks2...,
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
