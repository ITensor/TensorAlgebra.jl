using BlockArrays: AbstractBlockArray, AbstractBlockedUnitRange, Block, BlockedArray,
    blockedrange, blocklength, blocks, eachblockaxes1, mortar
using TensorAlgebra: TensorAlgebra, AbstractBlockTuple, BlockedTuple, FusionStyle,
    ReshapeFusion, matricize, matricize_axes, tensor_product_axis, unmatricize

struct BlockReshapeFusion <: FusionStyle end
TensorAlgebra.FusionStyle(::Type{<:AbstractBlockArray}) = BlockReshapeFusion()

function TensorAlgebra.trivial_axis(
        style::BlockReshapeFusion, side::Val{:codomain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return blockedrange([1])
end
function mortar_axis(axs)
    all(isone ∘ first, axs) ||
        throw(ArgumentError("Only one-based axes are supported"))
    return blockedrange(length.(axs))
end
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:codomain},
        r1::AbstractUnitRange, r2::AbstractUnitRange,
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = vec(map(splat(tensor_product_axis), blockaxpairs))
    return mortar_axis(blockaxs)
end
function TensorAlgebra.matricize(
        style::BlockReshapeFusion, a::AbstractArray, ndims_codomain::Val
    )
    ax = matricize_axes(style, a, ndims_codomain)
    reshaped_blocks_a = reshape(blocks(a), blocklength.(ax))
    bs = map(reshaped_blocks_a) do b
        matricize(b, ndims_codomain)
    end
    return mortar(bs, ax)
end
using BlockArrays: blocklengths
function TensorAlgebra.unmatricize(
        ::BlockReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    ax = (axes_codomain..., axes_domain...)
    reshaped_blocks_m = reshape(blocks(m), blocklength.(ax))
    bs = map(CartesianIndices(reshaped_blocks_m)) do I
        block_axes_I = BlockedTuple(
            map(ntuple(identity, length(ax))) do i
                return Base.axes1(ax[i][Block(I[i])])
            end,
            (length(axes_codomain), length(axes_domain)),
        )
        return unmatricize(reshaped_blocks_m[I], block_axes_I)
    end
    return mortar(bs, ax)
end

TensorAlgebra.FusionStyle(::Type{<:BlockedArray}) = ReshapeFusion()
unblock(a::BlockedArray) = a.blocks
unblock(a::AbstractBlockArray) = a[Base.OneTo.(size(a))...]
unblock(a::AbstractArray) = a
function TensorAlgebra.matricize(::ReshapeFusion, a::BlockedArray, ndims_codomain::Val)
    return matricize(ReshapeFusion(), unblock(a), ndims_codomain)
end
function unmatricize_blocked(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    a = unmatricize(
        ReshapeFusion(), m,
        Base.OneTo.(length.(axes_codomain)), Base.OneTo.(length.(axes_domain)),
    )
    return BlockedArray(a, (axes_codomain..., axes_domain...))
end
function TensorAlgebra.unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{AbstractBlockedUnitRange, Vararg{AbstractBlockedUnitRange}},
        axes_domain::Tuple{AbstractBlockedUnitRange, Vararg{AbstractBlockedUnitRange}},
    )
    return unmatricize_blocked(style, m, axes_codomain, axes_domain)
end
function TensorAlgebra.unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{AbstractBlockedUnitRange, Vararg{AbstractBlockedUnitRange}},
        axes_domain::Tuple{Vararg{AbstractBlockedUnitRange}},
    )
    return unmatricize_blocked(style, m, axes_codomain, axes_domain)
end
function TensorAlgebra.unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractBlockedUnitRange}},
        axes_domain::Tuple{AbstractBlockedUnitRange, Vararg{AbstractBlockedUnitRange}},
    )
    return unmatricize_blocked(style, m, axes_codomain, axes_domain)
end
