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

# add! for LinearBroadcasted subtypes.
function add!(dest::AbstractArray, src::ScaledBroadcasted, α::Number, β::Number)
    return add!(dest, unscaled(src), coeff(src) * α, β)
end

function add!(dest::AbstractArray, src::ConjBroadcasted, α::Number, β::Number)
    return permutedimsopadd!(dest, conj, unconj(src), ntuple(identity, ndims(dest)), α, β)
end

function add!(dest::AbstractArray, src::AddBroadcasted, α::Number, β::Number)
    args = addends(src)
    add!(dest, first(args), α, β)
    for a in Base.tail(args)
        add!(dest, a, α, true)
    end
    return dest
end

# add! for Mul materializes the factors and calls mul!.
function add!(dest::AbstractArray, src::Mul, α::Number, β::Number)
    return LA.mul!(dest, BC.materialize.(factors(src))..., α, β)
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
# Broadcast integration
# ---------------------------------------------------------------------------- #

broadcast_is_linear(f, args...) = false
broadcast_is_linear(::typeof(identity), ::Base.AbstractArrayOrBroadcasted) = true
broadcast_is_linear(::typeof(+), ::Base.AbstractArrayOrBroadcasted...) = true
broadcast_is_linear(::typeof(-), ::Base.AbstractArrayOrBroadcasted) = true
function broadcast_is_linear(
        ::typeof(-), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
broadcast_is_linear(::typeof(*), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
broadcast_is_linear(::typeof(\), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
broadcast_is_linear(::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
broadcast_is_linear(::typeof(/), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
function broadcast_is_linear(
        ::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return false
end
broadcast_is_linear(::typeof(*), ::Number, ::Number) = true
broadcast_is_linear(::typeof(conj), ::Base.AbstractArrayOrBroadcasted) = true
function broadcast_is_linear(
        ::Base.Fix1{typeof(*), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
function broadcast_is_linear(
        ::Base.Fix2{typeof(*), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
function broadcast_is_linear(
        ::Base.Fix2{typeof(/), <:Number}, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
# Check if a Broadcasted tree is linear and convert it to LinearBroadcasted
# in a single recursive pass. Returns `nothing` if nonlinear.
_to_linear(x) = x
function _to_linear(bc::BC.Broadcasted)
    broadcast_is_linear(bc.f, bc.args...) || return nothing
    args = map(_to_linear, bc.args)
    any(isnothing, args) && return nothing
    return linearbroadcasted(bc.f, args...)
end

"""
    broadcasted_linear(style, f, args...)

Validate that a broadcast expression is linear and convert it to a `LinearBroadcasted`
expression tree. Throws `ArgumentError` if the expression is not linear.

This is the entry point called by `BC.broadcasted(::LinearBroadcastedStyle, ...)` and
downstream broadcast styles that opt into linear broadcasting.
"""
function broadcasted_linear(style::BC.BroadcastStyle, f, args...)
    result = _to_linear(BC.Broadcasted(style, f, args))
    result === nothing && throw(
        ArgumentError(
            "Only linear broadcast operations are supported for `$style`, got `$f`."
        )
    )
    return result
end
function broadcasted_linear(f, args...)
    return broadcasted_linear(BC.combine_styles(args...), f, args...)
end

# Convert LinearBroadcasted / Mul back to Broadcasted for non-linear contexts.
to_broadcasted(x) = x
function to_broadcasted(a::AbstractArray)
    (BC.BroadcastStyle(typeof(a)) isa LinearBroadcastedStyle) || return a
    return BC.broadcasted(operation(a), to_broadcasted.(arguments(a))...)
end
function to_broadcasted(a::LinearBroadcasted)
    return BC.broadcasted(operation(a), to_broadcasted.(arguments(a))...)
end
# Matmul isn't a broadcasting operation so we materialize when building a
# broadcast expression involving a Mul.
to_broadcasted(a::Mul) = *(factors(a)...)
to_broadcasted(bc::BC.Broadcasted) = BC.Broadcasted(bc.f, to_broadcasted.(bc.args))

# LinearBroadcastedStyle for broadcast interop.
struct LinearBroadcastedStyle{N, Style <: BC.AbstractArrayStyle{N}} <:
    BC.AbstractArrayStyle{N}
    style::Style
end
# TODO: This empty constructor is required in some Julia versions below v1.12 (such as
# Julia v1.10), try deleting it once we drop support for those versions.
function LinearBroadcastedStyle{N, Style}() where {N, Style <: BC.AbstractArrayStyle{N}}
    return LinearBroadcastedStyle{N, Style}(Style())
end
function LinearBroadcastedStyle{N, Style}(
        ::Val{M}
    ) where {M, N, Style <: BC.AbstractArrayStyle{N}}
    return LinearBroadcastedStyle(Style(Val(M)))
end
function BC.BroadcastStyle(style1::LinearBroadcastedStyle, style2::LinearBroadcastedStyle)
    style = BC.BroadcastStyle(style1.style, style2.style)
    style ≡ BC.Unknown() && return BC.Unknown()
    return LinearBroadcastedStyle(style)
end
function Base.similar(bc::BC.Broadcasted{<:LinearBroadcastedStyle}, elt::Type, ax)
    return similar(BC.Broadcasted(bc.style.style, bc.f, bc.args, bc.axes), elt, ax)
end

# BroadcastStyle for LinearBroadcasted subtypes.
function BC.BroadcastStyle(::Type{<:ScaledBroadcasted{<:Any, A}}) where {A}
    return LinearBroadcastedStyle(BC.BroadcastStyle(A))
end
function BC.BroadcastStyle(::Type{<:ConjBroadcasted{A}}) where {A}
    return LinearBroadcastedStyle(BC.BroadcastStyle(A))
end
function BC.BroadcastStyle(::Type{<:AddBroadcasted{Args}}) where {Args}
    style = Base.promote_op(BC.combine_styles, fieldtypes(Args)...)()
    return LinearBroadcastedStyle(style)
end
function BC.BroadcastStyle(::Type{<:Mul{A, B}}) where {A, B}
    style = BC.BroadcastStyle(BC.BroadcastStyle(A), BC.BroadcastStyle(B))
    return LinearBroadcastedStyle(style)
end

# Broadcast.materialize for LinearBroadcasted and Mul.
BC.materialize(a::LinearBroadcasted) = copy(a)
BC.materialize(a::Mul) = copy(a)

# Backup definition: for broadcast operations that don't preserve lazy types
# (such as nonlinear operations), convert back to Broadcasted expressions.
function BC.broadcasted(::LinearBroadcastedStyle, f, args...)
    return BC.Broadcasted(f, to_broadcasted.(args))
end

# Linear broadcast operations produce LinearBroadcasted / Mul types.
function BC.broadcasted(
        ::LinearBroadcastedStyle,
        ::typeof(+),
        a::AbstractArray,
        b::AbstractArray
    )
    return linearbroadcasted(+, a, b)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle,
        ::typeof(+),
        a::AbstractArray,
        b::BC.Broadcasted
    )
    b_linear = _to_linear(b)
    b_linear === nothing && return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return linearbroadcasted(+, a, b_linear)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle,
        ::typeof(+),
        a::BC.Broadcasted,
        b::AbstractArray
    )
    a_linear = _to_linear(a)
    a_linear === nothing && return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return linearbroadcasted(+, a_linear, b)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle, ::typeof(+), a::BC.Broadcasted, b::BC.Broadcasted
    )
    return error("Not implemented")
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), α::Number, a::AbstractArray)
    return linearbroadcasted(*, α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), a::AbstractArray, α::Number)
    return linearbroadcasted(*, a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(\), α::Number, a::AbstractArray)
    return linearbroadcasted(\, α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(/), a::AbstractArray, α::Number)
    return linearbroadcasted(/, a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(-), a::AbstractArray)
    return linearbroadcasted(-, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(conj), a::AbstractArray)
    return linearbroadcasted(conj, a)
end
