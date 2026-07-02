module TensorAlgebraTensorKitExt

using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, ElementarySpace, ProductSpace,
    TensorMapWithStorage, numind, permute, project_symmetric!, space, spacetype,
    zerovector!, ←
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

# A plain-array prototype with native (space) axes is the operator/state construction case: `raw`
# is a dense matrix and the codomain/domain axes are `TensorMap` spaces. Build an uninitialized
# `TensorMap` whose block storage type `A` follows `raw`'s array type, so the storage stays on the
# same device (a GPU array's blocks stay on the GPU) while the map structure comes from the spaces.
# `A` is the return type of `similar(raw, T, ::Int)`, a 1-d vector of `raw`'s family, which is the
# block storage type. The elementary space type `S` is passed explicitly so the two dispatch
# entries below can read it from whichever of codomain/domain is non-empty and share one builder,
# mirroring `_map_homspace` and the map constructors.
function similar_tensormap(
        raw::AbstractArray, ::Type{T}, ::Type{S}, codomain_axes, domain_axes
    ) where {T, S <: ElementarySpace}
    A = Base.promote_op(similar, typeof(raw), Type{T}, Int)
    return TensorMapWithStorage{T, A}(undef, _map_homspace(S, codomain_axes, domain_axes))
end
function TensorAlgebra.similar_map(
        raw::AbstractArray, ::Type{T},
        codomain_axes::Tuple{S, Vararg{S}}, domain_axes::Tuple{Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return similar_tensormap(raw, T, S, codomain_axes, domain_axes)
end
function TensorAlgebra.similar_map(
        raw::AbstractArray, ::Type{T},
        codomain_axes::Tuple{}, domain_axes::Tuple{S, Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return similar_tensormap(raw, T, S, codomain_axes, domain_axes)
end

# ===============================  zeros_map / randn_map / rand_map  ========================
# A `TensorMap` keeps its codomain and domain as separate `ProductSpace`s rather than a single
# flattened axis, so build the `codomain ← domain` space directly instead of the dense
# flatten-and-dualize fallback. As with `similar_map`, the axes arrive codomain-facing
# (un-dualized), which is TensorKit's own codomain/domain convention. The elementary space type
# `S` is passed to `_map_homspace` explicitly so the two dispatch entries per constructor can
# read it from whichever of the codomain/domain is non-empty and share one builder; an empty
# axis tuple gives the unit space `ProductSpace{S}()`.
function _map_homspace(::Type{S}, codomain_axes, domain_axes) where {S <: ElementarySpace}
    return ProductSpace{S}(codomain_axes...) ← ProductSpace{S}(domain_axes...)
end
function TensorAlgebra.zeros_map(
        ::Type{T}, codomain_axes::Tuple{S, Vararg{S}}, domain_axes::Tuple{Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return TensorKit.zeros(T, _map_homspace(S, codomain_axes, domain_axes))
end
function TensorAlgebra.zeros_map(
        ::Type{T}, codomain_axes::Tuple{}, domain_axes::Tuple{S, Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return TensorKit.zeros(T, _map_homspace(S, codomain_axes, domain_axes))
end
for (f, g) in ((:randn_map, :randn), (:rand_map, :rand))
    @eval begin
        function TensorAlgebra.$f(
                rng::AbstractRNG, ::Type{T},
                codomain_axes::Tuple{S, Vararg{S}}, domain_axes::Tuple{Vararg{S}}
            ) where {T, S <: ElementarySpace}
            return TensorKit.$g(rng, T, _map_homspace(S, codomain_axes, domain_axes))
        end
        function TensorAlgebra.$f(
                rng::AbstractRNG, ::Type{T},
                codomain_axes::Tuple{}, domain_axes::Tuple{S, Vararg{S}}
            ) where {T, S <: ElementarySpace}
            return TensorKit.$g(rng, T, _map_homspace(S, codomain_axes, domain_axes))
        end
    end
end

# =====================================  projectto!  ========================================
# `projectto!` places dense `src` data into the restricted (symmetric) space of `dest`. A
# `TensorMap` is not an `AbstractArray`, so the generic `copyto!` default does not apply; delegate
# to TensorKit's `project_symmetric!`, which fills the symmetry-allowed blocks from the dense data
# and discards any component outside the block structure. Composed with the map constructors above,
# this makes `project(dense, codomain_axes, domain_axes)` build a `TensorMap` from a dense matrix.
function TensorAlgebra.projectto!(dest::AbstractTensorMap, src::AbstractArray)
    return project_symmetric!(dest, src)
end

# The generic `checked_projectto!` verifies the projection with `isapprox(src, dest)`, but a
# `TensorMap` `dest` is not elementwise-comparable to the dense `src`. Densify `dest` with
# `convert(Array, ...)` so the check is the same elementwise `isapprox(src, dest)` as the dense path,
# keeping one `InexactError`/`kwargs` contract across backends rather than TensorKit's own
# residual-norm `tol`/`ArgumentError` check.
function TensorAlgebra.checked_projectto!(
        dest::AbstractTensorMap,
        src::AbstractArray;
        kwargs...
    )
    TensorAlgebra.projectto!(dest, src)
    isapprox(src, convert(Array, dest); kwargs...) ||
        throw(InexactError(:checked_projectto!, typeof(dest), src))
    return dest
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

# ==================================  linear-combination broadcast  =========================
# A `TensorMap` is not an `AbstractArray`, so it needs a `BroadcastStyle` to broadcast lazily
# (otherwise Base tries to `collect` it). A linear combination flattens (via `tryflattenlinear`)
# to a `LinearBroadcasted` that materializes through `add!`/`bipermutedimsopadd!` above; the
# `copyto!` here is not piracy because `LinearBroadcasted` is TensorAlgebra-owned. Element-wise
# (nonlinear) broadcast is not a meaningful operation on a symmetric tensor, so it errors rather
# than dense-converting.
struct TensorMapStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:AbstractTensorMap}) = TensorMapStyle()
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::TensorMapStyle) = s
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::Base.Broadcast.BroadcastStyle) = s
Base.Broadcast.broadcastable(a::AbstractTensorMap) = a

function Base.copyto!(dest::AbstractTensorMap, src::TensorAlgebra.LinearBroadcasted)
    return TensorAlgebra.add!(dest, src, true, false)
end

function Base.copy(::Base.Broadcast.Broadcasted{TensorMapStyle})
    return error(
        "element-wise broadcast is not supported for a `TensorMap`; only linear combinations \
        such as `a .+ b` and `2 .* a` are supported"
    )
end

end
