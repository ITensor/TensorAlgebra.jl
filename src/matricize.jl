using LinearAlgebra: Diagonal

using BlockArrays: AbstractBlockedUnitRange, blockedrange

using TensorProducts: ⊗
using .BaseExtensions: _permutedims, _permutedims!

# =====================================  FusionStyle  ======================================
abstract type FusionStyle end

FusionStyle(x) = FusionStyle(typeof(x))
FusionStyle(T::Type) = throw(MethodError(FusionStyle, (T,)))

# =======================================  misc  ========================================
trivial_axis(::Tuple{}) = Base.OneTo(1)
trivial_axis(::Tuple{Vararg{AbstractUnitRange}}) = Base.OneTo(1)
trivial_axis(::Tuple{Vararg{AbstractBlockedUnitRange}}) = blockedrange([1])

# Inner version takes a list of sub-permutations, overload this one if needed.
function fuseaxes(
        axes::Tuple{Vararg{AbstractUnitRange}}, lengths::Val...
    )
    axesblocks = blocks(axes[blockedtrivialperm(lengths)])
    return map(block -> isempty(block) ? trivial_axis(axes) : ⊗(block...), axesblocks)
end

# Inner version takes a list of sub-permutations, overload this one if needed.
function fuseaxes(
        axes::Tuple{Vararg{AbstractUnitRange}}, permblocks::Tuple{Vararg{Int}}...
    )
    axes′ = map(d -> axes[d], permmortar(permblocks))
    return fuseaxes(axes′, Val.(length.(permblocks))...)
end

function fuseaxes(
        axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockPermutation
    )
    return fuseaxes(axes, blocks(blockedperm)...)
end

# Inner version takes a list of sub-permutations, overload this one if needed.
function permuteblockeddims(a::AbstractArray, perm1, perm2)
    return _permutedims(a, (perm1..., perm2...))
end
function permuteblockeddims!(a_dest::AbstractArray, a_src::AbstractArray, perm1, perm2)
    return _permutedims!(a_dest, a_src, (perm1..., perm2...))
end

# TODO remove _permutedims once support for Julia 1.10 is dropped
# define permutedims with a BlockedPermuation. Default is to flatten it.
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

function matricize(a::AbstractArray, length1::Val, length2::Val)
    return matricize(FusionStyle(a), a, length1, length2)
end
# This is the primary function that should be overloaded for new fusion styles.
# This assumes the permutation was already performed.
function matricize(style::FusionStyle, a::AbstractArray, length1::Val, length2::Val)
    return throw(
        MethodError(
            matricize, Tuple{typeof(style), typeof(a), typeof(length1), typeof(length2)}
        )
    )
end

function matricize(
        a::AbstractArray, permblock1::Tuple{Vararg{Int}}, permblock2::Tuple{Vararg{Int}}
    )
    return matricize(FusionStyle(a), a, permblock1, permblock2)
end
# This is a more advanced version to overload where the permutation is actually performed.
function matricize(
        style::FusionStyle, a::AbstractArray,
        permblock1::NTuple{N1, Int}, permblock2::NTuple{N2, Int}
    ) where {N1, N2}
    ndims(a) == length(permblock1) + length(permblock2) ||
        throw(ArgumentError("Invalid bipermutation"))
    a_perm = permuteblockeddims(a, permblock1, permblock2)
    return matricize(style, a_perm, Val(length(permblock1)), Val(length(permblock2)))
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
function matricize(a::AbstractArray, permblock1, permblock2)
    return matricize(FusionStyle(a), a, permblock1, permblock2)
end
function matricize(style::FusionStyle, a::AbstractArray, permblock1, permblock2)
    return matricize(style, a, to_permblocks(a, (permblock1, permblock2))...)
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
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return unmatricize(FusionStyle(m), m, codomain_axes, domain_axes)
end
# This is the primary function that should be overloaded for new fusion styles.
function unmatricize(
        style::FusionStyle, m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return throw(
        MethodError(
            unmatricize,
            Tuple{
                typeof(style), typeof(m), typeof(codomain_axes), typeof(domain_axes),
            },
        )
    )
end

function unmatricize(
        m::AbstractMatrix,
        blocked_axes::BlockedTuple{2, <:Any, <:Tuple{Vararg{AbstractUnitRange}}},
    )
    return unmatricize(FusionStyle(m), m, blocked_axes)
end
function unmatricize(
        style::FusionStyle,
        m::AbstractMatrix,
        blocked_axes::BlockedTuple{2, <:Any, <:Tuple{Vararg{AbstractUnitRange}}},
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
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
    )
    invbiperm = permmortar((invperm1, invperm2))
    length(axes_dest) == length(invbiperm) ||
        throw(ArgumentError("axes do not match permutation"))
    blocked_axes = axes_dest[invbiperm]
    a12 = unmatricize(style, m, blocked_axes)
    biperm_dest = biperm(invperm(invbiperm), length_codomain(axes_dest))
    return permuteblockeddims(a12, biperm_dest)
end

function unmatricize(m::AbstractMatrix, axes_dest, invbiperm::AbstractBlockPermutation{2})
    return unmatricize(FusionStyle(m), m, axes_dest, invbiperm)
end
function unmatricize(
        style::FusionStyle, m::AbstractMatrix, axes_dest,
        invbiperm::AbstractBlockPermutation{2}
    )
    return unmatricize(style, m, axes_dest, blocks(invbiperm)...)
end

function unmatricize!(
        a_dest::AbstractArray, m::AbstractMatrix,
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
    )
    return unmatricize!(FusionStyle(m), a_dest, m, invperm1, invperm2)
end
function unmatricize!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
    )
    invbiperm = permmortar((invperm1, invperm2))
    ndims(a_dest) == length(invbiperm) ||
        throw(ArgumentError("destination does not match permutation"))
    blocked_axes = axes(a_dest)[invbiperm]
    a_perm = unmatricize(style, m, blocked_axes)
    biperm_dest = biperm(invperm(invbiperm), length_codomain(axes(a_dest)))
    return permuteblockeddims!(a_dest, a_perm, biperm_dest)
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
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
        α::Number, β::Number
    )
    return unmatricizeadd!(FusionStyle(a_dest), a_dest, m, invperm1, invperm2, α, β)
end
function unmatricizeadd!(
        style::FusionStyle, a_dest::AbstractArray, m::AbstractMatrix,
        invperm1::Tuple{Vararg{Int}}, invperm2::Tuple{Vararg{Int}},
        α::Number, β::Number,
    )
    a12 = unmatricize(style, m, axes(a_dest), invperm1, invperm2)
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
function matricize(style::ReshapeFusion, a::AbstractArray, length1::Val, length2::Val)
    return reshape(a, fuseaxes(axes(a), length1, length2))
end
function unmatricize(
        style::ReshapeFusion, m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return reshape(m, (codomain_axes..., domain_axes...))
end
