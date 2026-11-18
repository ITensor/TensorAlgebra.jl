module TensorAlgebraTensorKitExt

using MatrixAlgebraKit: diagview
using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, DiagonalTensorMap, ElementarySpace,
    ProductSpace, TensorMap, TensorMapWithStorage, blocks, codomain, dim, domain, dual,
    fuse, numind, numout, permute, project_symmetric!, sectors, space, spacetype, ←
using TensorOperations: TensorOperations as TO
using VectorInterface: VectorInterface

# ============================  AbstractArray-vocabulary bridge  ============================
# The `i`-th axis of a `TensorMap` is its `i`-th index space `space(t, i)`, which is already
# dualized for domain indices.
TensorAlgebra.ndims(t::AbstractTensorMap) = numind(t)
TensorAlgebra.axes(t::AbstractTensorMap, i::Int) = space(t, i)
TensorAlgebra.axes(t::AbstractTensorMap) = ntuple(i -> space(t, i), numind(t))
# A `TensorMap` has no `Base.size`; its dense size is the per-index space dimension.
TensorAlgebra.size(t::AbstractTensorMap, i::Int) = dim(space(t, i))
TensorAlgebra.size(t::AbstractTensorMap) = ntuple(i -> dim(space(t, i)), numind(t))
TensorAlgebra.ndims_codomain(t::AbstractTensorMap) = numout(t)
TensorAlgebra.hasbipartition(::AbstractTensorMap) = true

# `t[]` on a rank-0 `TensorMap` requires a trivial sector type; `TensorKit.scalar` is the
# general spelling.
TensorAlgebra.scalar(t::AbstractTensorMap) = TensorKit.scalar(t)

# The generic `data` follows `Base.parent`, which does not reach a `TensorMap`'s flat storage
# vector. A general `AbstractTensorMap` has no single storage vector, so it gets no method.
TensorAlgebra.data(t::TensorMap) = TensorAlgebra.data(t.data)
TensorAlgebra.data(t::DiagonalTensorMap) = TensorAlgebra.data(t.data)

TensorAlgebra.isdual(V::ElementarySpace) = TensorKit.isdual(V)
TensorAlgebra.dual(V::ElementarySpace) = TensorKit.dual(V)

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

# The ungraded extent of a space-backed axis is the range over its dense dimension, dropping
# sectors and the arrow so `conj`-equal spaces share an ungraded value.
TensorAlgebra.ungrade(V::ElementarySpace) = Base.OneTo(dim(V))

# Sum of the dense elements. Through the dense presentation rather than the block data:
# for a non-abelian sector type the dense embedding expands each block by its fusion-tree
# structure, so the block-data sum would differ.
TensorAlgebra.sum(t::AbstractTensorMap; kwargs...) = Base.sum(convert(Array, t); kwargs...)

# =====================================  similar_map  =======================================
# `similar_map` takes the domain axes un-dualized, which matches TensorKit's
# `similar(t, T, codomain, domain)`.
function TensorAlgebra.similar_map(
        a::AbstractTensorMap, ::Type{T}, axes_codomain, axes_domain
    ) where {T}
    S = spacetype(a)
    return similar(a, T, ProductSpace{S}(axes_codomain...), ProductSpace{S}(axes_domain...))
end

# A plain-array prototype with space axes builds a `TensorMap` whose block storage type follows
# `raw`'s array type, so the storage stays on the same device, for example a GPU.
function similar_tensormap(
        raw::AbstractArray, ::Type{T}, ::Type{S}, axes_codomain, axes_domain
    ) where {T, S <: ElementarySpace}
    A = Base.promote_op(similar, typeof(raw), Type{T}, Int)
    return TensorMapWithStorage{T, A}(undef, _map_homspace(S, axes_codomain, axes_domain))
end
function TensorAlgebra.similar_map(
        raw::AbstractArray, ::Type{T},
        axes_codomain::Tuple{S, Vararg{S}}, axes_domain::Tuple{Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return similar_tensormap(raw, T, S, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        raw::AbstractArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{S, Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return similar_tensormap(raw, T, S, axes_codomain, axes_domain)
end

# ===============================  zeros_map / randn_map / rand_map  ========================
# Build the `codomain ← domain` space directly. `S` is passed explicitly since an empty axis
# tuple, which gives the unit space `ProductSpace{S}()`, does not carry it.
function _map_homspace(::Type{S}, axes_codomain, axes_domain) where {S <: ElementarySpace}
    return ProductSpace{S}(axes_codomain...) ← ProductSpace{S}(axes_domain...)
end
function TensorAlgebra.zeros_map(
        ::Type{T}, axes_codomain::Tuple{S, Vararg{S}}, axes_domain::Tuple{Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return TensorKit.zeros(T, _map_homspace(S, axes_codomain, axes_domain))
end
function TensorAlgebra.zeros_map(
        ::Type{T}, axes_codomain::Tuple{}, axes_domain::Tuple{S, Vararg{S}}
    ) where {T, S <: ElementarySpace}
    return TensorKit.zeros(T, _map_homspace(S, axes_codomain, axes_domain))
end
for (f, g) in ((:randn_map, :randn), (:rand_map, :rand))
    @eval begin
        function TensorAlgebra.$f(
                rng::AbstractRNG, ::Type{T},
                axes_codomain::Tuple{S, Vararg{S}}, axes_domain::Tuple{Vararg{S}}
            ) where {T, S <: ElementarySpace}
            return TensorKit.$g(rng, T, _map_homspace(S, axes_codomain, axes_domain))
        end
        function TensorAlgebra.$f(
                rng::AbstractRNG, ::Type{T},
                axes_codomain::Tuple{}, axes_domain::Tuple{S, Vararg{S}}
            ) where {T, S <: ElementarySpace}
            return TensorKit.$g(rng, T, _map_homspace(S, axes_codomain, axes_domain))
        end
    end
end

# =====================================  projectto!  ========================================
# `project_symmetric!` requires a matching dense size, so `src` is reshaped first, which lets it
# omit trailing length-1 axes like the generic `projectto!`.
function TensorAlgebra.projectto!(dest::AbstractTensorMap, src::AbstractArray)
    TensorAlgebra.check_project_size(size(src), TensorAlgebra.size(dest))
    return project_symmetric!(dest, reshape(src, TensorAlgebra.size(dest)))
end

# A `TensorMap` carries its own split, so `unproject` is `convert(Array, t)`, only checking the
# caller's split matches the map's codomain rank.
function TensorAlgebra.unproject(t::AbstractTensorMap, ::Val{K}) where {K}
    K == numout(t) || throw(
        ArgumentError(
            "`unproject`: codomain split Val($K) does not match the `TensorMap`'s codomain rank $(numout(t))"
        )
    )
    return convert(Array, t)
end

# =============================  infer_aux_space (project_aux derivation)  ===================
# Two entries so `S` can be read from whichever of the codomain and domain is non-empty.
function TensorAlgebra.infer_aux_space(
        raw::AbstractArray, axes_codomain::Tuple{S, Vararg{S}}, axes_domain::Tuple{Vararg{S}}
    ) where {S <: ElementarySpace}
    return infer_aux_space_tensormap(raw, S, axes_codomain, axes_domain)
end
function TensorAlgebra.infer_aux_space(
        raw::AbstractArray, axes_codomain::Tuple{}, axes_domain::Tuple{S, Vararg{S}}
    ) where {S <: ElementarySpace}
    return infer_aux_space_tensormap(raw, S, axes_codomain, axes_domain)
end

# Candidate sectors come from `codomain ⊗ conj(domain)` in sorted order, since a `GradedSpace`
# sorts its sectors and the dense layout follows. The result may span several sectors.
function infer_aux_space_tensormap(
        raw, ::Type{S}, axes_codomain, axes_domain
    ) where {S <: ElementarySpace}
    aux_dim = length(axes_codomain) + length(axes_domain) + 1
    aux_length = size(raw, aux_dim)
    content = fuse(axes_codomain..., dual.(axes_domain)...)
    # This builds one `TensorMap` per candidate column. Reading each column's sector directly from
    # the block structure of the fused content could avoid that.
    function slice_is_covariant(r, s)
        slice = selectdim(raw, aux_dim, r)
        return !isnothing(
            TensorAlgebra.tryproject(slice, axes_codomain, (axes_domain..., S(s => 1)))
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

# ===============================  bipermutedimsconjadd!  ====================================
# Delegate to TensorKit's TensorOperations interface: `tensoradd!` realizes the permutation, the
# data conjugation (via `adjoint` internally), and the `α`/`β` scaling in one call.
function TensorAlgebra.bipermutedimsconjadd!(
        dest::AbstractTensorMap, isconj::Bool, src::AbstractTensorMap,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    TO.tensoradd!(dest, src, (perm_codomain, perm_domain), isconj, α, β)
    return dest
end

# ==================================  matricize / unmatricize  ==============================
# A `TensorMap` is already a linear map, so matricizing it regroups its indices with `permute`.

# `permute` at the tensor's own codomain/domain split is trivial and returns `t` itself, so that
# split is the one memory-sharing matricization (TensorKit's own `has_shared_permute` notion). Any
# other split, or a folded `conj`, regroups into a fresh `TensorMap`.
function TensorAlgebra.isoutputview(
        ::typeof(TensorAlgebra.matricizeconj), isconj::Bool,
        t::AbstractTensorMap, perm_codomain, perm_domain
    )
    return !isconj &&
        TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain) &&
        length(perm_codomain) == numout(t)
end
function TensorAlgebra.matricizeconjview(
        isconj::Bool, t::AbstractTensorMap, perm_codomain, perm_domain
    )
    return t
end
# A `TensorMap`'s matricization is a regrouping of its indices, so its destination is the permuted
# one, whose spaces are dualized under `isconj` as `tensoradd!` requires.
function TensorAlgebra.allocate_output(
        ::typeof(TensorAlgebra.matricizeconj), isconj::Bool,
        t::AbstractTensorMap, perm_codomain, perm_domain
    )
    return TensorAlgebra.allocate_output(
        TensorAlgebra.bipermutedimsconj, isconj, t, perm_codomain, perm_domain
    )
end
function TensorAlgebra.matricizeconj!(
        dest::AbstractTensorMap, isconj::Bool,
        t::AbstractTensorMap, perm_codomain, perm_domain
    )
    return TensorAlgebra.bipermutedimsconjadd!(
        dest, isconj, t, perm_codomain, perm_domain, true, false
    )
end

# TensorKit owns its own `one!` generic and forwards only the `AbstractMatrix` case to
# MatrixAlgebraKit, so the identity fill on a regrouped `TensorMap` goes through TensorKit.
TensorAlgebra.MatrixAlgebra.one!(t::AbstractTensorMap) = TensorKit.one!(t)

function TensorAlgebra.unmatricize(m::AbstractTensorMap, axes_codomain, axes_domain)
    S = spacetype(m)
    dest = ProductSpace{S}(axes_codomain...) ← ProductSpace{S}(axes_domain...)
    space(m) == dest ||
        throw(ArgumentError("`unmatricize` space `$dest` does not match `$(space(m))`"))
    return m
end

# ======================================  contract  =========================================
# `TensorMap`s contract through TensorKit's TensorOperations interface, selected by
# `default_algorithm` below.
TensorAlgebra.zero!(t::AbstractTensorMap) = VectorInterface.zerovector!(t)

# A `TensorMap` is not an `AbstractArray`, so the generic in-place operations do not apply.
TensorAlgebra.scale!(t::AbstractTensorMap, β::Number) = VectorInterface.scale!(t, β)
function TensorAlgebra.scaleadd!(
        y::AbstractTensorMap, x::AbstractTensorMap, α::Number, β::Number
    )
    return VectorInterface.add!(y, x, α, β)
end

function TensorAlgebra.default_algorithm(
        ::typeof(TensorAlgebra.contract!),
        ::Type{<:Tuple{AbstractTensorMap, AbstractTensorMap, AbstractTensorMap}}
    )
    return TensorAlgebra.ContractAlgorithm(TO.DefaultBackend())
end

# ==================================  linear-combination broadcast  =========================
# A `TensorMap` needs a `BroadcastStyle` to broadcast lazily, otherwise Base tries to `collect`
# it. Only linear combinations are supported, since an elementwise nonlinear broadcast is not
# meaningful on a symmetric tensor.
struct TensorMapStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:AbstractTensorMap}) = TensorMapStyle()
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::TensorMapStyle) = s
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::Base.Broadcast.BroadcastStyle) = s
Base.Broadcast.broadcastable(a::AbstractTensorMap) = a

function Base.copyto!(dest::AbstractTensorMap, src::TensorAlgebra.LinearBroadcasted)
    return TensorAlgebra.scaleadd!(dest, src, true, false)
end

# Allocation for a linear-combination `copy`/`copyto!`: seed the result off a `TensorMap` operand
# (`broadcast_prototype`) so it inherits that operand's storage type, and take the axes from the
# flattened expression, which carry the conj-dualization that Base's `combine_axes` drops.
function Base.similar(bc::Base.Broadcast.Broadcasted{TensorMapStyle}, ::Type{T}) where {T}
    lb = TensorAlgebra.flattenlinear(bc)
    return TensorAlgebra.similar_map(broadcast_prototype(lb), T, axes(lb), ())
end
# The first `TensorMap` operand in a broadcast expression, unwrapping the linear-fold leaves.
broadcast_prototype(a::AbstractTensorMap) = a
broadcast_prototype(a::TensorAlgebra.PermutedDims) = broadcast_prototype(parent(a))
function broadcast_prototype(a::TensorAlgebra.ScaledBroadcasted)
    return broadcast_prototype(TensorAlgebra.unscaled(a))
end
broadcast_prototype(a::TensorAlgebra.ConjBroadcasted) = broadcast_prototype(parent(a))
function broadcast_prototype(a::TensorAlgebra.AddBroadcasted)
    return broadcast_prototype(first(TensorAlgebra.addends(a)))
end

function Base.copy(bc::Base.Broadcast.Broadcasted{TensorMapStyle})
    lb = TensorAlgebra.tryflattenlinear(bc)
    isnothing(lb) && error(
        "element-wise broadcast is not supported for a `TensorMap`; only linear combinations \
        such as `a .+ b` and `2 .* a` are supported"
    )
    return copy(lb)
end

# ====================================  pow_diag_safe  ======================================
# `MAK.diagview` of a `TensorMap` is shape-shifting (a `SectorVector` for a `DiagonalTensorMap`,
# a per-sector dict otherwise), so clamp the diagonal per block instead: each block's `diagview`
# is a plain vector view regardless of the map's type.
function TensorAlgebra.MatrixAlgebra.pow_diag_safe!(
        Dp::AbstractTensorMap, D::AbstractTensorMap, p, tol
    )
    for ((_, bp), (_, b)) in zip(blocks(Dp), blocks(D))
        map!(
            d -> TensorAlgebra.MatrixAlgebra._clamped_pow(d, p, tol),
            diagview(bp),
            diagview(b)
        )
    end
    return Dp
end

end
