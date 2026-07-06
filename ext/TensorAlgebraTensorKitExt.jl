module TensorAlgebraTensorKitExt

using MatrixAlgebraKit: diagview
using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, ElementarySpace, ProductSpace,
    TensorMapWithStorage, blocks, codomain, dim, domain, dual, fuse, numind, permute,
    project_symmetric!, sectors, space, spacetype, zerovector!, ←
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
# A `TensorMap` has no `Base.size`; its dense size is the per-index space dimension.
TensorAlgebra.size(t::AbstractTensorMap, i::Int) = dim(space(t, i))
TensorAlgebra.size(t::AbstractTensorMap) = ntuple(i -> dim(space(t, i)), numind(t))

# TensorKit deliberately defines no `Base.conj` for tensors; its native conjugation is
# `adjoint`, which also swaps the codomain and domain. `TensorAlgebra.conjugate`
# (conjugated elements on dualized spaces, leg order unchanged) is the adjoint followed by
# the block swap returning each leg to its original flat position — the inverse of
# TensorKit's `adjointtensorindex` convention (`i <= numout ? numin + i : i - numout`),
# the same lowering its TensorOperations interface uses for `conj` arguments.
function TensorAlgebra.conjugate(t::AbstractTensorMap)
    m, n = TensorKit.numout(t), TensorKit.numin(t)
    p = ntuple(i -> i <= m ? n + i : i - m, m + n)
    return permute(adjoint(t), (p[1:m], p[(m + 1):(m + n)]))
end

# `t[]` on a rank-0 `TensorMap` requires a trivial sector type; `TensorKit.scalar` is the
# general spelling.
TensorAlgebra.scalar(t::AbstractTensorMap) = TensorKit.scalar(t)

# The trivial length-1 axis of a space is its unit space (`oneunit`), the trivial-sector
# one-dimensional space; the length-`n` form is the direct sum of `n` unit spaces.
TensorAlgebra.trivialrange(V::ElementarySpace) = oneunit(V)
TensorAlgebra.trivialrange(::Type{S}) where {S <: ElementarySpace} = oneunit(S)
function TensorAlgebra.trivialrange(V::ElementarySpace, n::Integer)
    return TensorAlgebra.trivialrange(typeof(V), n)
end
function TensorAlgebra.trivialrange(::Type{S}, n::Integer) where {S <: ElementarySpace}
    return TensorKit.oplus(ntuple(Returns(oneunit(S)), n)...)
end

# Sum of the dense elements. Through the dense presentation rather than the block data:
# for a non-abelian sector type the dense embedding expands each block by its fusion-tree
# structure, so the block-data sum would differ.
TensorAlgebra.sum(t::AbstractTensorMap; kwargs...) = Base.sum(convert(Array, t); kwargs...)

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
# `project_symmetric!` requires a matching dense size, so reshape `src` to `size(dest)` first (a
# no-op when the ranks already match); this lets a lower-rank `src` omit trailing length-1 axes,
# matching the generic `projectto!`.
function TensorAlgebra.projectto!(dest::AbstractTensorMap, src::AbstractArray)
    return project_symmetric!(dest, reshape(src, TensorAlgebra.size(dest)))
end

# The `is_projected` check compares through `convert(Array, dest)`, which TensorKit already
# defines for an `AbstractTensorMap`, so no dedicated method is needed here.

# =============================  allocate_project (aux-leg derivation)  =====================
# `allocate_project` for `TensorMap` spaces. With no surplus axis this is plain `similar_map`;
# a single trailing surplus axis in `raw` is an auxiliary leg whose space is derived (see
# `infer_aux_space`) and appended as the last domain axis so the result is symmetry-allowed.
function TensorAlgebra.allocate_project(
        raw::AbstractArray, codomain_axes::Tuple{S, Vararg{S}}, domain_axes::Tuple{Vararg{S}}
    ) where {S <: ElementarySpace}
    nphys = length(codomain_axes) + length(domain_axes)
    ndims(raw) <= nphys &&
        return TensorAlgebra.similar_map(raw, codomain_axes, domain_axes)
    ndims(raw) == nphys + 1 || throw(
        ArgumentError(
            "`project`: expected at most one trailing auxiliary axis beyond the $nphys \
            given axes, got a rank-$(ndims(raw)) input"
        )
    )
    aux = infer_aux_space(raw, codomain_axes, domain_axes)
    return TensorAlgebra.similar_map(raw, codomain_axes, (domain_axes..., aux))
end

# The space of `raw`'s trailing auxiliary axis, derived so the projected result is
# symmetry-allowed. Candidates are the operator content `codomain ⊗ conj(domain)`, scanned in
# canonical (sorted) sector order — a `GradedSpace` sorts its sectors and the dense layout
# follows, so the aux slices must appear in that order. The result may span several sectors (a
# direct-sum, MPO-style virtual leg).
function infer_aux_space(
        raw, codomain_axes::Tuple{S, Vararg{S}}, domain_axes::Tuple{Vararg{S}}
    ) where {S <: ElementarySpace}
    aux_dim = length(codomain_axes) + length(domain_axes) + 1
    aux_length = size(raw, aux_dim)
    content = fuse(codomain_axes..., dual.(domain_axes)...)
    # Probe the surplus axis slice by slice: a slice keeps the aux axis (width `dim(s)`), so its
    # rank matches the candidate axes exactly and `tryproject` allocates, fills, and round-trip-
    # verifies without re-entering the derivation branch. This builds one `TensorMap` per candidate
    # column, which is fine for operator-sized inputs but not cheap. A faster derivation would read
    # the covariant sectors from the fusion-tree block structure (as the `TensorMap` constructor
    # does) instead of constructing a projection per slice.
    function slice_is_covariant(r, s)
        slice = selectdim(raw, aux_dim, r)
        return !isnothing(
            TensorAlgebra.tryproject(slice, codomain_axes, (domain_axes..., S(s => 1)))
        )
    end
    seccounts = Pair{TensorKit.sectortype(S), Int}[]
    pos = 1
    for s in sectors(content)
        d = dim(S(s => 1))
        m = 0
        while pos + d - 1 <= aux_length && slice_is_covariant(pos:(pos + d - 1), s)
            m += 1
            pos += d
        end
        m > 0 && push!(seccounts, s => m)
    end
    pos == aux_length + 1 || throw(
        ArgumentError(
            "`project`: could not derive a covariant auxiliary space for the surplus axis of \
            length $aux_length; the aux slices must be ordered by the canonical (sorted) sector \
            order of the derived space"
        )
    )
    return S(seccounts...)
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

# The identity fill on the regrouped map is TensorKit's own `one!` (MatrixAlgebraKit's
# `one!` speaks `AbstractMatrix` only).
function TensorAlgebra.one!!(
        style::TensorKitFusion, A::AbstractTensorMap, ndims_codomain::Val; kwargs...
    )
    return TensorKit.one!(TensorAlgebra.matricize(style, A, ndims_codomain))
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

# ====================================  pow_diag_safe  ======================================
# `MAK.diagview` of a `TensorMap` is shape-shifting (a `SectorVector` for a `DiagonalTensorMap`,
# a per-sector dict otherwise), so clamp the diagonal per block instead: each block's `diagview`
# is a plain vector view regardless of the map's type.
function TensorAlgebra.MatrixAlgebra._pow_diag!(D::AbstractTensorMap, p, tol)
    for (_, b) in blocks(D)
        σ = diagview(b)
        map!(d -> TensorAlgebra.MatrixAlgebra._clamped_pow(d, p, tol), σ, σ)
    end
    return D
end

end
