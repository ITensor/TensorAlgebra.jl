using BlockArrays: AbstractBlockArray, AbstractBlockedUnitRange, blockedrange,
    eachblockaxes1

struct BlockReshapeFusion <: FusionStyle end

FusionStyle(::Type{<:AbstractBlockArray}) = BlockReshapeFusion()

trivial_axis(::Type{<:AbstractBlockedUnitRange}) = blockedrange([1])
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
    return reshape(a, matricize_axes(style, a, ndims_codomain))
end
