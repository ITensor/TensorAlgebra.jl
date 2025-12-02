using LinearAlgebra: Diagonal

using BlockArrays: AbstractBlockedUnitRange, blockedrange

using TensorProducts: ⊗
using .BaseExtensions: _permutedims, _permutedims!

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

FusionStyle(x) = FusionStyle(typeof(x))
FusionStyle(T::Type) = throw(MethodError(FusionStyle, (T,)))

# =======================================  misc  ========================================
axis_type(a::AbstractArray) = eltype(axes(a))
axis_type(a::AbstractArray{<:Any, 0}) = Base.OneTo{Int}
trivial_axis(::Type{<:AbstractUnitRange}) = Base.OneTo(1)
trivial_axis(::Type{<:AbstractBlockedUnitRange}) = blockedrange([1])

# Fallback to `TensorProducts.⊗`.
# TODO: Remove dependency on TensorProducts.jl, update downstream packages:
# BlockSparseArrays.jl, GradedArrays.jl, FusionTensors.jl.
tensor_product(::FusionStyle, ax1, ax2) = ax1 ⊗ ax2

# Inner version takes a list of sub-permutations, overload this one if needed.
function matricize_axes(style::FusionStyle, a::AbstractArray, length_codomain::Val)
    unval(length_codomain) ≤ ndims(a) ||
        throw(ArgumentError("Codomain length exceeds number of dimensions."))
    biperm = trivialbiperm(length_codomain, Val(ndims(a)))
    axesblocks = blocks(axes(a)[biperm])
    init_axis = trivial_axis(axis_type(a))
    return map(axesblocks) do axesblock
        return reduce(axesblock; init = init_axis) do ax1, ax2
            return tensor_product(style, ax1, ax2)
        end
    end
end
function matricize_axes(a::AbstractArray, length_codomain::Val)
    return matricize_axes(FusionStyle(a), a, length_codomain)
end

# Inner version takes a list of sub-permutations, overload this one if needed.
# TODO: Remove _permutedims once support for Julia 1.10 is dropped
# define permutedims with a BlockedPermuation. Default is to flatten it.
# TODO: Deprecate `permuteblockeddims` in favor of `bipermutedims`.
# Keeping it here for backwards compatibility.
function bipermutedims(a::AbstractArray, perm1, perm2)
    return permuteblockeddims(a, perm1, perm2)
end
function bipermutedims!(a_dest::AbstractArray, a_src::AbstractArray, perm1, perm2)
    return permuteblockeddims!(a_dest, a_src, perm1, perm2)
end
function bipermutedims(a::AbstractArray, biperm::AbstractBlockPermutation{2})
    return bipermutedims(a, blocks(biperm)...)
end
function bipermutedims!(
        a_dest::AbstractArray, a_src::AbstractArray, biperm::AbstractBlockPermutation{2}
    )
    return bipermutedims!(a_dest, a_src, blocks(biperm)...)
end

# Older interface.
# TODO: Deprecate in favor of `bipermutedims` (or decide if we want to keep it
# in case there are applications of more general partitionings).
function permuteblockeddims(a::AbstractArray, perm1, perm2)
    return _permutedims(a, (perm1..., perm2...))
end
function permuteblockeddims!(a_dest::AbstractArray, a_src::AbstractArray, perm1, perm2)
    return _permutedims!(a_dest, a_src, (perm1..., perm2...))
end
function permuteblockeddims(a::AbstractArray, biperm::AbstractBlockPermutation{2})
    return permuteblockeddims(a, blocks(biperm)...)
end
function permuteblockeddims!(
        a_dest::AbstractArray, a_src::AbstractArray, biperm::AbstractBlockPermutation{2}
    )
    return permuteblockeddims!(a_dest, a_src, blocks(biperm)...)
end

# =====================================  matricize  ========================================
# TBD settle copy/not copy convention
# matrix factorizations assume copy
# maybe: copy=false kwarg

function matricize(a::AbstractArray, length_codomain::Val)
    return matricize(FusionStyle(a), a, length_codomain)
end
# This is the primary function that should be overloaded for new fusion styles.
# This assumes the permutation was already performed.
function matricize(
        style::FusionStyle, a::AbstractArray, length_codomain::Val
    )
    return throw(
        MethodError(
            matricize, Tuple{typeof(style), typeof(a), typeof(length_codomain)}
        )
    )
end

function matricize(
        a::AbstractArray,
        permblock_codomain::Tuple{Vararg{Int}}, permblock_domain::Tuple{Vararg{Int}}
    )
    return matricize(FusionStyle(a), a, permblock_codomain, permblock_domain)
end
# This is a more advanced version to overload where the permutation is actually performed.
function matricize(
        style::FusionStyle, a::AbstractArray,
        permblock_codomain::Tuple{Vararg{Int}}, permblock_domain::Tuple{Vararg{Int}}
    )
    ndims(a) == length(permblock_codomain) + length(permblock_domain) ||
        throw(ArgumentError("Invalid bipermutation"))
    a_perm = bipermutedims(a, permblock_codomain, permblock_domain)
    return matricize(style, a_perm, Val(length(permblock_codomain)))
end

# Process inputs such as `EllipsisNotation.Ellipsis`.
function to_permblocks(a::AbstractArray, permblocks::NTuple{2, Tuple{Vararg{Int}}})
    isperm((permblocks[1]..., permblocks[2]...)) ||
        throw(ArgumentError("Invalid bipermutation"))
    return permblocks
end
# Like `setcomplement` is like `setdiff` but assumes t2 ⊆ t1.
function tuplesetcomplement(t1::NTuple{N1}, t2::NTuple{N2}) where {N1, N2}
    t2 ⊆ t1 || throw(ArgumentError("t2 must be a subset of t1"))
    return NTuple{N1 - N2}(setdiff(t1, t2))
end
function to_permblocks(
        a::AbstractArray, permblocks::Tuple{Tuple{Ellipsis}, Tuple{Vararg{Int}}}
    )
    permblocks1 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[2])
    return (permblocks1, permblocks[2])
end
function to_permblocks(
        a::AbstractArray, permblocks::Tuple{Tuple{Vararg{Int}}, Tuple{Ellipsis}}
    )
    permblocks2 = tuplesetcomplement(ntuple(identity, ndims(a)), permblocks[1])
    return (permblocks[1], permblocks2)
end

function matricize(a::AbstractArray, permblock_codomain, permblock_domain)
    return matricize(FusionStyle(a), a, permblock_codomain, permblock_domain)
end
function matricize(
        style::FusionStyle, a::AbstractArray, permblock_codomain, permblock_domain
    )
    return matricize(style, a, to_permblocks(a, (permblock_codomain, permblock_domain))...)
end

function matricize(a::AbstractArray, biperm_dest::AbstractBlockPermutation{2})
    return matricize(FusionStyle(a), a, biperm_dest)
end
function matricize(
        style::FusionStyle, a::AbstractArray, biperm_dest::AbstractBlockPermutation{2}
    )
    return matricize(style, a, blocks(biperm_dest)...)
end

# ====================================  unmatricize  =======================================
function unmatricize(
        m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return unmatricize(FusionStyle(m), m, axes_codomain, axes_domain)
end
# This is the primary function that should be overloaded for new fusion styles.
function unmatricize(
        style::FusionStyle, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return throw(
        MethodError(
            unmatricize,
            Tuple{
                typeof(style), typeof(m), typeof(axes_codomain), typeof(axes_domain),
            },
        )
    )
end

function unmatricize(m::AbstractMatrix, blocked_axes::AbstractBlockTuple{2})
    return unmatricize(FusionStyle(m), m, blocked_axes)
end
function unmatricize(
        style::FusionStyle, m::AbstractMatrix, blocked_axes::AbstractBlockTuple{2}
    )
    return unmatricize(style, m, blocks(blocked_axes)...)
end

function unmatricize(
        m::AbstractMatrix, axes_dest,
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
    )
    return unmatricize(FusionStyle(m), m, axes_dest, invperm1, invperm2)
end
function unmatricize(
        style::FusionStyle, m::AbstractMatrix, axes_dest,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
    )
    invbiperm = permmortar((invperm_codomain, invperm_domain))
    length(axes_dest) == length(invbiperm) ||
        throw(ArgumentError("axes do not match permutation"))
    blocked_axes = axes_dest[invbiperm]
    a12 = unmatricize(style, m, blocked_axes)
    biperm_dest = biperm(invperm(invbiperm), length_codomain(axes_dest))
    return bipermutedims(a12, biperm_dest)
end

function unmatricize(m::AbstractMatrix, axes_dest, invbiperm::AbstractBlockPermutation{2})
    return unmatricize(FusionStyle(m), m, axes_dest, invbiperm)
end
function unmatricize(
        style::FusionStyle, m::AbstractMatrix, axes_dest,
        invbiperm::AbstractBlockPermutation{2},
    )
    return unmatricize(style, m, axes_dest, blocks(invbiperm)...)
end

function unmatricize!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
    )
    return unmatricize!(FusionStyle(m), a_dest, m, invperm_codomain, invperm_domain)
end
function unmatricize!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
    )
    invbiperm = permmortar((invperm_codomain, invperm_domain))
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    blocked_axes = axes(a_dest)[invbiperm]
    a_perm = unmatricize(style, m, blocked_axes)
    biperm_dest = biperm(invperm(invbiperm), length_codomain(axes(a_dest)))
    return bipermutedims!(a_dest, a_perm, biperm_dest)
end

function unmatricize!(
        a_dest::AbstractArray, m::AbstractMatrix, invbiperm::AbstractBlockPermutation{2}
    )
    return unmatricize!(FusionStyle(m), a_dest, m, invbiperm)
end
function unmatricize!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invbiperm::AbstractBlockPermutation{2},
    )
    return unmatricize!(style, a_dest, m, blocks(invbiperm)...)
end

function unmatricizeadd!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
        α::Number, β::Number
    )
    return unmatricizeadd!(
        FusionStyle(a_dest), a_dest, m, invperm_codomain, invperm_domain, α, β
    )
end
function unmatricizeadd!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
        α::Number, β::Number,
    )
    a12 = unmatricize(style, m, axes(a_dest), invperm_codomain, invperm_domain)
    a_dest .= α .* a12 .+ β .* a_dest
    return a_dest
end

function unmatricizeadd!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invbiperm::AbstractBlockPermutation{2},
        α::Number, β::Number
    )
    return unmatricizeadd!(FusionStyle(a_dest), a_dest, m, invbiperm, α, β)
end
function unmatricizeadd!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invbiperm::AbstractBlockPermutation{2},
        α::Number, β::Number,
    )
    return unmatricizeadd!(
        style, a_dest, m, blocks(invbiperm)..., α, β
    )
end

# Defaults to ReshapeFusion, a simple reshape
struct ReshapeFusion <: FusionStyle end
FusionStyle(::Type{<:AbstractArray}) = ReshapeFusion()
function matricize(style::ReshapeFusion, a::AbstractArray, length_codomain::Val)
    return reshape(a, matricize_axes(style, a, length_codomain))
end
function unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return reshape(m, (axes_codomain..., axes_domain...))
end
