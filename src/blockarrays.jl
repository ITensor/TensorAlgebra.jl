using BlockArrays: AbstractBlockArray, AbstractBlockedUnitRange, BlockedArray, blockedrange,
    eachblockaxes1, mortar

struct BlockReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:AbstractBlockArray}) = BlockReshapeFusion()

trivial_axis(::BlockReshapeFusion) = blockedrange([1])
function mortar_axis(axs)
    all(isone ∘ first, axs) ||
        throw(ArgumentError("Only one-based axes are supported"))
    return blockedrange(length.(axs))
end
function tensor_product_axis(
        ::BlockReshapeFusion, r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    isone(first(r1)) || isone(first(r2)) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = vec(map(splat(tensor_product_axis), blockaxpairs))
    return mortar_axis(blockaxs)
end
function matricize(style::BlockReshapeFusion, a::AbstractArray, ndims_codomain::Val)
    ax = matricize_axes(style, a, ndims_codomain)
    reshaped_blocks_a = reshape(blocks(a), blocklength.(ax))
    bs = map(reshaped_blocks_a) do b
        matricize(b, ndims_codomain)
    end
    return mortar(bs, ax)
end
using BlockArrays: blocklengths
function unmatricize(
        ::BlockReshapeFusion,
        m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    ax = (codomain_axes..., domain_axes...)
    reshaped_blocks_m = reshape(blocks(m), blocklength.(ax))
    bs = map(CartesianIndices(reshaped_blocks_m)) do I
        block_axes_I = BlockedTuple(
            map(ntuple(identity, length(ax))) do i
                return Base.axes1(ax[i][Block(I[i])])
            end,
            (length(codomain_axes), length(domain_axes)),
        )
        return unmatricize(reshaped_blocks_m[I], block_axes_I)
    end
    return mortar(bs, ax)
end

struct BlockedReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:BlockedArray}) = BlockedReshapeFusion()
unblock(a::BlockedArray) = a.blocks
unblock(a::AbstractBlockArray) = a[Base.OneTo.(size(a))...]
unblock(a::AbstractArray) = a
function matricize(::BlockedReshapeFusion, a::AbstractArray, ndims_codomain::Val)
    return matricize(ReshapeFusion(), unblock(a), ndims_codomain)
end
function unmatricize(
        style::BlockedReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    a = unmatricize(
        ReshapeFusion(), m,
        Base.OneTo.(length.(axes_codomain)), Base.OneTo.(length.(axes_domain)),
    )
    return BlockedArray(a, (axes_codomain..., axes_domain...))
end
