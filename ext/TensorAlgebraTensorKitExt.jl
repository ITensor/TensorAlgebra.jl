module TensorAlgebraTensorKitExt

using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, ProductSpace, numind, permute, space,
    spacetype, zerovector!, ←
using TensorOperations: TensorOperations as TO

# ============================  AbstractArray-vocabulary bridge  ============================
# TensorAlgebra's generic orchestration describes operands in the `AbstractArray` vocabulary
# (`TensorAlgebra.ndims`, `TensorAlgebra.axes`), while a `TensorMap` speaks `numind`/`space`.
# Overload the TensorAlgebra-owned accessors so a `TensorMap` flows through the generic code
# unchanged. The `i`-th "axis" of a `TensorMap` is its `i`-th index space `space(t, i)`, which
# for domain indices is already dualized.
TensorAlgebra.ndims(t::AbstractTensorMap) = numind(t)
TensorAlgebra.axes(t::AbstractTensorMap, i::Int) = space(t, i)
TensorAlgebra.axes(t::AbstractTensorMap) = ntuple(i -> space(t, i), numind(t))

# =====================================  similar_map  =======================================
# `similar_map` takes the codomain/domain axes in codomain-facing (un-dualized) form, which is
# exactly what TensorKit's `similar(t, T, codomain, domain)` wants, so build the two
# `ProductSpace`s directly.
function TensorAlgebra.similar_map(
        a::AbstractTensorMap, ::Type{T}, codomain_axes, domain_axes
    ) where {T}
    S = spacetype(a)
    return similar(a, T, ProductSpace{S}(codomain_axes...), ProductSpace{S}(domain_axes...))
end

# ================================  bipermutedimsopadd!  =====================================
# `dest = β * dest + α * permutedims(op.(src), (perm_codomain, perm_domain))`. Delegate to
# TensorKit's TensorOperations interface: `tensoradd!` realizes the permutation, the `op === conj`
# data conjugation (via `adjoint` internally), and the `α`/`β` scaling in one call.
function TensorAlgebra.bipermutedimsopadd!(
        dest::AbstractTensorMap, op, src::AbstractTensorMap,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    conjA = op === conj
    (op === identity || conjA) ||
        throw(ArgumentError("`op` must be `identity` or `conj`, got `$op`"))
    TO.tensoradd!(dest, src, (perm_codomain, perm_domain), conjA, α, β)
    return dest
end

# ==================================  matricize / unmatricize  ==============================
# A `TensorMap` is already a linear map codomain ← domain, so "matricizing" is just regrouping
# its indices into the requested codomain/domain bipartition (`permute`). No fusion or copy of
# the array vocabulary is needed: MatrixAlgebraKit factorizes the regrouped `TensorMap` directly.
struct TensorKitFusion <: TensorAlgebra.FusionStyle end
TensorAlgebra.FusionStyle(::Type{<:AbstractTensorMap}) = TensorKitFusion()

function TensorAlgebra.matricize(
        ::TensorKitFusion, t::AbstractTensorMap, ndims_codomain::Val{K}
    ) where {K}
    N = numind(t)
    return permute(t, (ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))))
end

# `unmatricize` reconstructs the codomain/domain axes from the matrix `m`. A `TensorMap` already
# is the linear map its space describes, so the only valid request is the one whose codomain/domain
# split matches `m`'s own space, and `unmatricize` returns `m` unchanged. The domain axes arrive
# codomain-facing (un-dualized), which is exactly TensorKit's domain convention, so they build the
# domain `ProductSpace` directly.
function TensorAlgebra.unmatricize(
        ::TensorKitFusion, m::AbstractTensorMap, codomain_axes, domain_axes
    )
    S = spacetype(m)
    dest = ProductSpace{S}(codomain_axes...) ← ProductSpace{S}(domain_axes...)
    space(m) == dest ||
        throw(ArgumentError("`unmatricize` space `$dest` does not match `$(space(m))`"))
    return m
end

# ======================================  contract  =========================================
# Contraction of `TensorMap`s is index regrouping plus a matrix product, which TensorKit
# already implements through its TensorOperations interface. Route the generic `contract`
# there: `zero!` clears the `similar_map`-allocated destination, and the default algorithm
# hands the in-place contraction to the TensorOperations backend (see the TensorOperations
# extension's `contractopadd!`).
TensorAlgebra.zero!(t::AbstractTensorMap) = zerovector!(t)

function TensorAlgebra.default_contract_algorithm(
        ::Type{<:AbstractTensorMap}, ::Type{<:AbstractTensorMap}
    )
    return TensorAlgebra.ContractAlgorithm(TO.DefaultBackend())
end

end
