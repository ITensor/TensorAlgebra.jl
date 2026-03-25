import Base.Broadcast as BC
import LinearAlgebra as LA

# TermInterface-like interface.
iscall(x) = false
function operation end
function arguments end

# ---------------------------------------------------------------------------- #
# LinearBroadcasted — lazy linear broadcast expressions (not <: AbstractArray)
# ---------------------------------------------------------------------------- #

"""
    LinearBroadcasted

Abstract supertype for lazy linear broadcast expressions. Analogous to
`Base.Broadcast.Broadcasted` but restricted to linear operations.

Materializes via the protocol:
copy(lb) = copyto!(similar(lb), lb)
copyto!(dest, lb) → add!(dest, lb, 1, 0)
"""
abstract type LinearBroadcasted end

# Generic interface for LinearBroadcasted subtypes.
Base.axes(a::LinearBroadcasted, d::Int) = axes(a)[d]
Base.similar(a::LinearBroadcasted) = similar(a, eltype(a))
Base.similar(a::LinearBroadcasted, elt::Type) = similar(a, elt, axes(a))
function Base.show(io::IO, a::LinearBroadcasted)
    print(io, operation(a), "(", join(arguments(a), ", "), ")")
    return nothing
end
iscall(::LinearBroadcasted) = true

# --- ScaledBroadcasted --------------------------------------------------------

struct ScaledBroadcasted{C <: Number, A} <: LinearBroadcasted
    coeff::C
    parent::A
end

unscaled(a::ScaledBroadcasted) = a.parent
coeff(a::ScaledBroadcasted) = a.coeff

Base.axes(a::ScaledBroadcasted) = axes(unscaled(a))
function Base.eltype(a::ScaledBroadcasted)
    return Base.promote_op(*, typeof(coeff(a)), eltype(unscaled(a)))
end
Base.ndims(a::ScaledBroadcasted) = ndims(unscaled(a))

function Base.similar(a::ScaledBroadcasted, elt::Type, ax)
    return similar(unscaled(a), elt, ax)
end

operation(::ScaledBroadcasted) = *
arguments(a::ScaledBroadcasted) = (coeff(a), unscaled(a))

# --- ConjBroadcasted ----------------------------------------------------------

struct ConjBroadcasted{A} <: LinearBroadcasted
    parent::A
end

unconj(a::ConjBroadcasted) = a.parent

Base.axes(a::ConjBroadcasted) = axes(unconj(a))
Base.eltype(a::ConjBroadcasted) = eltype(unconj(a))
Base.ndims(a::ConjBroadcasted) = ndims(unconj(a))

function Base.similar(a::ConjBroadcasted, elt::Type, ax)
    return similar(unconj(a), elt, ax)
end

operation(::ConjBroadcasted) = conj
arguments(a::ConjBroadcasted) = (unconj(a),)

# --- AddBroadcasted -----------------------------------------------------------

struct AddBroadcasted{Args <: Tuple} <: LinearBroadcasted
    args::Args
    AddBroadcasted(args...) = new{typeof(args)}(args)
end

addends(a::AddBroadcasted) = a.args

Base.axes(a::AddBroadcasted) = BC.combine_axes(addends(a)...)
Base.eltype(a::AddBroadcasted) = Base.promote_op(+, eltype.(addends(a))...)
Base.ndims(a::AddBroadcasted) = ndims(first(addends(a)))

function Base.similar(a::AddBroadcasted, elt::Type, ax)
    return similar(BC.Broadcasted(+, addends(a)), elt, ax)
end

operation(::AddBroadcasted) = +
arguments(a::AddBroadcasted) = addends(a)

# ---------------------------------------------------------------------------- #
# Mul — lazy matrix multiplication (standalone, not LinearBroadcasted)
# ---------------------------------------------------------------------------- #

# Same as `LinearAlgebra.matprod`, but duplicated here since it is private.
matprod(x, y) = x * y + x * y

struct Mul{A, B}
    a::A
    b::B
end

factors(a::Mul) = (a.a, a.b)

Base.axes(a::Mul) = (axes(a.a, 1), axes(a.b, ndims(a.b)))
Base.axes(a::Mul, d::Int) = axes(a)[d]
Base.eltype(a::Mul) = Base.promote_op(matprod, eltype(a.a), eltype(a.b))
Base.ndims(a::Mul) = ndims(a.b)
Base.size(a::Mul) = length.(axes(a))

Base.similar(a::Mul) = similar(a, eltype(a))
Base.similar(a::Mul, elt::Type) = similar(a, elt, axes(a))
function Base.similar(a::Mul, elt::Type, ax)
    return similar(BC.materialize(last(factors(a))), elt, ax)
end

function Base.show(io::IO, a::Mul)
    f = factors(a)
    print(io, "*(", f[1], ", ", f[2], ")")
    return nothing
end

iscall(::Mul) = true
operation(::Mul) = *
arguments(a::Mul) = factors(a)

# ---------------------------------------------------------------------------- #
# Materialization protocol: copy, copyto!, add!
# ---------------------------------------------------------------------------- #

function Base.copy(a::LinearBroadcasted)
    return copyto!(similar(a), a)
end

function Base.copy(a::Mul)
    return copyto!(similar(a), a)
end

# copyto! for LinearBroadcasted dispatches to add!.
function Base.copyto!(dest::AbstractArray, src::LinearBroadcasted)
    return add!(dest, src, true, false)
end

# copyto! for Mul dispatches to mul!. Materialize factors first since
# they may be LinearBroadcasted types.
function Base.copyto!(dest::AbstractArray, src::Mul)
    return LA.mul!(dest, BC.materialize.(factors(src))...)
end

# Op composition with simplification rules.
_compose_op(::typeof(identity), g) = g
_compose_op(f, ::typeof(identity)) = f
_compose_op(::typeof(identity), ::typeof(identity)) = identity
_compose_op(::typeof(conj), ::typeof(conj)) = identity
_compose_op(f, g) = f ∘ g

# permutedimsopadd! for LinearBroadcasted subtypes.
function permutedimsopadd!(
        dest::AbstractArray, op, src::ScaledBroadcasted, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, op, unscaled(src), perm, op(coeff(src)) * α, β)
end

function permutedimsopadd!(
        dest::AbstractArray, op, src::ConjBroadcasted, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, _compose_op(op, conj), unconj(src), perm, α, β)
end

function permutedimsopadd!(
        dest::AbstractArray, op, src::AddBroadcasted, perm, α::Number, β::Number
    )
    args = addends(src)
    permutedimsopadd!(dest, op, first(args), perm, α, β)
    for a in Base.tail(args)
        permutedimsopadd!(dest, op, a, perm, α, true)
    end
    return dest
end

# TODO: Replace with contractopadd! once that interface exists,
# to avoid materializing the Mul intermediate.
function permutedimsopadd!(
        dest::AbstractArray, op, src::Mul, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, op, copy(src), perm, α, β)
end

# ---------------------------------------------------------------------------- #
# linearbroadcasted — construct LinearBroadcasted subtypes by dispatching on f
# ---------------------------------------------------------------------------- #

"""
    linearbroadcasted(f, args...)

Construct a `LinearBroadcasted` subtype from function `f` and arguments.
Analogous to `Base.Broadcast.broadcasted(f, args...)`.

# Examples

```julia
linearbroadcasted(*, 2.0, a)   # ScaledBroadcasted(2.0, a)
linearbroadcasted(conj, a)     # ConjBroadcasted(a)
linearbroadcasted(+, a, b)     # AddBroadcasted(a, b)
```
"""
function linearbroadcasted end

# Scaling: Number * AbstractArray
linearbroadcasted(::typeof(*), α::Number, a::AbstractArray) = ScaledBroadcasted(α, a)
linearbroadcasted(::typeof(*), a::AbstractArray, α::Number) = ScaledBroadcasted(α, a)
# Scaling of ScaledBroadcasted: absorb coefficient.
function linearbroadcasted(::typeof(*), α::Number, a::ScaledBroadcasted)
    return ScaledBroadcasted(α * coeff(a), unscaled(a))
end

# Conjugation.
linearbroadcasted(::typeof(conj), a::AbstractArray) = ConjBroadcasted(a)
linearbroadcasted(::typeof(conj), a::AbstractArray{<:Real}) = a
linearbroadcasted(::typeof(conj), a::ConjBroadcasted) = unconj(a)
function linearbroadcasted(::typeof(conj), a::ScaledBroadcasted)
    return ScaledBroadcasted(conj(coeff(a)), linearbroadcasted(conj, unscaled(a)))
end

# Addition.
linearbroadcasted(::typeof(+), a, b) = AddBroadcasted(a, b)
function linearbroadcasted(f::typeof(+), a, b, c, xs...)
    return Base.afoldl(
        (x, y) -> linearbroadcasted(f, x, y),
        linearbroadcasted(f, linearbroadcasted(f, a, b), c),
        xs...
    )
end
# Flatten AddBroadcasted + anything.
linearbroadcasted(::typeof(+), a::AddBroadcasted, b) = AddBroadcasted(addends(a)..., b)
linearbroadcasted(::typeof(+), a, b::AddBroadcasted) = AddBroadcasted(a, addends(b)...)
function linearbroadcasted(::typeof(+), a::AddBroadcasted, b::AddBroadcasted)
    return AddBroadcasted(addends(a)..., addends(b)...)
end
linearbroadcasted(::typeof(+), a) = a

# Subtraction.
linearbroadcasted(::typeof(-), a, b) = linearbroadcasted(+, a, linearbroadcasted(*, -1, b))
linearbroadcasted(::typeof(-), a) = linearbroadcasted(*, -1, a)

# Division / left-division by scalars.
linearbroadcasted(::typeof(/), a, b::Number) = linearbroadcasted(*, inv(b), a)
linearbroadcasted(::typeof(\), a::Number, b) = linearbroadcasted(*, inv(a), b)

# Identity.
linearbroadcasted(::typeof(identity), a) = a

# Fix1/Fix2 wrappers for scalar multiplication/division.
linearbroadcasted(f::Base.Fix1{typeof(*)}, a) = linearbroadcasted(*, f.x, a)
linearbroadcasted(f::Base.Fix2{typeof(*)}, a) = linearbroadcasted(*, a, f.x)
linearbroadcasted(f::Base.Fix2{typeof(/)}, a) = linearbroadcasted(/, a, f.x)

# Scaling of AddBroadcasted distributes.
function linearbroadcasted(::typeof(*), α::Number, a::AddBroadcasted)
    return linearbroadcasted(+, map(x -> linearbroadcasted(*, α, x), addends(a))...)
end

# Conjugation of AddBroadcasted distributes.
function linearbroadcasted(::typeof(conj), a::AddBroadcasted)
    return linearbroadcasted(+, map(x -> linearbroadcasted(conj, x), addends(a))...)
end

# Conjugation of Mul distributes.
function linearbroadcasted(::typeof(conj), a::Mul)
    f = factors(a)
    return Mul(linearbroadcasted(conj, f[1]), linearbroadcasted(conj, f[2]))
end

# Scaling of Mul: wrap in ScaledBroadcasted.
linearbroadcasted(::typeof(*), α::Number, a::Mul) = ScaledBroadcasted(α, a)

# Number * Number passthrough (for broadcast lowering).
linearbroadcasted(::typeof(*), a::Number, b::Number) = a * b

# ---------------------------------------------------------------------------- #
# Broadcast integration — instantiation-time conversion
# ---------------------------------------------------------------------------- #

"""
    islinearbroadcast(f, args...) -> Bool

Per-node trait: can `(f, args...)` be expressed as a `LinearBroadcasted`?
Extensible by downstream packages for additional linear operations.
"""
islinearbroadcast(f, args...) = false
islinearbroadcast(::typeof(identity), ::Base.AbstractArrayOrBroadcasted) = true
islinearbroadcast(::typeof(+), ::Base.AbstractArrayOrBroadcasted...) = true
islinearbroadcast(::typeof(-), ::Base.AbstractArrayOrBroadcasted) = true
function islinearbroadcast(
        ::typeof(-), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
islinearbroadcast(::typeof(*), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
islinearbroadcast(::typeof(\), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
islinearbroadcast(::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
islinearbroadcast(::typeof(/), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
function islinearbroadcast(
        ::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return false
end
islinearbroadcast(::typeof(*), ::Number, ::Number) = true
islinearbroadcast(::typeof(conj), ::Base.AbstractArrayOrBroadcasted) = true
function islinearbroadcast(
        ::Base.Fix1{typeof(*), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
function islinearbroadcast(
        ::Base.Fix2{typeof(*), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
function islinearbroadcast(
        ::Base.Fix2{typeof(/), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end

"""
    tryflattenlinear(bc::Broadcasted) -> LinearBroadcasted or nothing

Recursively convert a `Broadcasted` tree to a `LinearBroadcasted` tree.
Returns `nothing` if any node is not linear (as determined by `islinearbroadcast`).

Analogous to `Broadcast.flatten` for `Broadcasted` trees, but converts to
`LinearBroadcasted` subtypes via `linearbroadcasted`.

Downstream styles call this from `Base.copy(::Broadcasted{MyStyle})` to
opt into linear broadcasting at materialization time.
"""
tryflattenlinear(x) = x
function tryflattenlinear(bc::BC.Broadcasted)
    islinearbroadcast(bc.f, bc.args...) || return nothing
    args = map(tryflattenlinear, bc.args)
    any(isnothing, args) && return nothing
    return linearbroadcasted(bc.f, args...)
end

# BroadcastStyle for LinearBroadcasted subtypes — delegate to the wrapped array type.
function BC.BroadcastStyle(::Type{<:ScaledBroadcasted{<:Any, A}}) where {A}
    return BC.BroadcastStyle(A)
end
function BC.BroadcastStyle(::Type{<:ConjBroadcasted{A}}) where {A}
    return BC.BroadcastStyle(A)
end
function BC.BroadcastStyle(::Type{<:AddBroadcasted{Args}}) where {Args}
    return Base.promote_op(BC.combine_styles, fieldtypes(Args)...)()
end
function BC.BroadcastStyle(::Type{<:Mul{A, B}}) where {A, B}
    return BC.BroadcastStyle(BC.BroadcastStyle(A), BC.BroadcastStyle(B))
end

# Broadcast.materialize for LinearBroadcasted and Mul.
BC.materialize(a::LinearBroadcasted) = copy(a)
BC.materialize(a::Mul) = copy(a)
