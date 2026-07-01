module TensorAlgebraTensorKitExt

using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, ProductSpace, TensorMap, codomain, domain,
    dual, numind, permute, scalartype, space, spacetype, zerovector!, ←, ≅
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

# `unmatricize` reconstructs the codomain/domain axes from the matrix `m`. When the axes match `m`
# index-for-index (the common case, since `matricize` only regroups) it returns `m` unchanged. More
# generally the axes may *fuse* to `m`'s codomain/domain (an ITensor-style combiner split): a fused
# split is a reshape, not a basis change, so when each group is isomorphic to `m`'s the block data
# carries over unchanged and is rewrapped on the new spaces with no copy. The domain axes arrive
# dualized (index-space convention), so `dual` recovers the stored domain.
function TensorAlgebra.unmatricize(
        ::TensorKitFusion, m::AbstractTensorMap, codomain_axes, domain_axes
    )
    dest_codomain = ProductSpace(codomain_axes...)
    dest_domain = ProductSpace(map(dual, domain_axes)...)
    space(m) == (dest_codomain ← dest_domain) && return m
    (codomain(m) ≅ dest_codomain && domain(m) ≅ dest_domain) || throw(
        ArgumentError(
            "`unmatricize` axes `$(dest_codomain ← dest_domain)` do not fuse to `$(space(m))`"
        )
    )
    return TensorMap{scalartype(m)}(m.data, dest_codomain ← dest_domain)
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
