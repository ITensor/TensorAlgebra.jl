import Base.Broadcast as BC
import FunctionImplementations as FI
import LinearAlgebra as LA
import StridedViews as SV

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

function Base.adjoint(a::ScaledBroadcasted)
    return ScaledBroadcasted(coeff(a), adjoint(unscaled(a)))
end
function Base.transpose(a::ScaledBroadcasted)
    return ScaledBroadcasted(coeff(a), transpose(unscaled(a)))
end

function FI.permuteddims(a::ScaledBroadcasted, perm)
    return ScaledBroadcasted(coeff(a), FI.permuteddims(unscaled(a), perm))
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

Base.conj(a::ConjBroadcasted) = unconj(a)
Base.adjoint(a::ConjBroadcasted) = transpose(unconj(a))
Base.transpose(a::ConjBroadcasted) = adjoint(unconj(a))

function FI.permuteddims(a::ConjBroadcasted, perm)
    return ConjBroadcasted(FI.permuteddims(unconj(a), perm))
end

SV.isstrided(a::ConjBroadcasted) = SV.isstrided(unconj(a))
SV.StridedView(a::ConjBroadcasted) = conj(SV.StridedView(unconj(a)))

operation(::ConjBroadcasted) = conj
arguments(a::ConjBroadcasted) = (unconj(a),)

# --- AddBroadcasted -----------------------------------------------------------

struct AddBroadcasted{Args <: Tuple} <: LinearBroadcasted
    args::Args
    function AddBroadcasted(args...)
        if !allequal(ndims, args)
            error("All addends must have the same number of dimensions.")
        end
        return new{typeof(args)}(args)
    end
end

addends(a::AddBroadcasted) = a.args

Base.axes(a::AddBroadcasted) = BC.combine_axes(addends(a)...)
Base.eltype(a::AddBroadcasted) = Base.promote_op(+, eltype.(addends(a))...)
Base.ndims(a::AddBroadcasted) = ndims(first(addends(a)))

function Base.similar(a::AddBroadcasted, elt::Type, ax)
    return similar(BC.Broadcasted(+, addends(a)), elt, ax)
end

function Base.adjoint(a::AddBroadcasted)
    return AddBroadcasted(adjoint.(addends(a))...)
end
function Base.transpose(a::AddBroadcasted)
    return AddBroadcasted(transpose.(addends(a))...)
end

function FI.permuteddims(a::AddBroadcasted, perm)
    return AddBroadcasted(Base.Fix2(FI.permuteddims, perm).(addends(a))...)
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

function Base.adjoint(a::Mul)
    f = factors(a)
    return Mul(adjoint(f[2]), adjoint(f[1]))
end
function Base.transpose(a::Mul)
    f = factors(a)
    return Mul(transpose(f[2]), transpose(f[1]))
end

function FI.permuteddims(a::Mul, perm)
    perm == (1, 2) && return a
    return transpose(a)
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
    return add!(dest, conj(unconj(src)), α, β)
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
# LinearBroadcastFunction — constructor API
# ---------------------------------------------------------------------------- #

"""
    LinearBroadcastFunction(f)

Wrap a function `f` so that calling it produces a `LinearBroadcasted` expression
instead of eagerly computing. Analogous to `Base.BroadcastFunction`.

# Examples

```julia
LinearBroadcastFunction(*)(2.0, a)   # ScaledBroadcasted(2.0, a)
LinearBroadcastFunction(conj)(a)     # ConjBroadcasted(a)
LinearBroadcastFunction(+)(a, b)     # AddBroadcasted(a, b)
```
"""
struct LinearBroadcastFunction{F} <: Function
    f::F
end

"""
    LinearFunc

Shorthand for `LinearBroadcastFunction`.

# Examples

```julia
LinearFunc(*)(2.0, a)   # ScaledBroadcasted(2.0, a)
LinearFunc(conj)(a)     # ConjBroadcasted(a)
LinearFunc(+)(a, b)     # AddBroadcasted(a, b)
```
"""
const LinearFunc = LinearBroadcastFunction

# Scaling: Number * AbstractArray
function (::typeof(LinearFunc(*)))(α::Number, a::AbstractArray)
    return ScaledBroadcasted(α, a)
end
function (::typeof(LinearFunc(*)))(a::AbstractArray, α::Number)
    return ScaledBroadcasted(α, a)
end
# Scaling of ScaledBroadcasted: absorb coefficient.
function (::typeof(LinearFunc(*)))(α::Number, a::ScaledBroadcasted)
    return ScaledBroadcasted(α * coeff(a), unscaled(a))
end

# Conjugation.
function (::typeof(LinearFunc(conj)))(a::AbstractArray)
    return ConjBroadcasted(a)
end
(::typeof(LinearFunc(conj)))(a::AbstractArray{<:Real}) = a
(::typeof(LinearFunc(conj)))(a::ConjBroadcasted) = unconj(a)
function (::typeof(LinearFunc(conj)))(a::ScaledBroadcasted)
    return ScaledBroadcasted(
        conj(coeff(a)), LinearFunc(conj)(unscaled(a))
    )
end

# Addition.
function (lf::typeof(LinearFunc(+)))(a, b)
    return AddBroadcasted(a, b)
end
function (lf::typeof(LinearFunc(+)))(a, b, c, xs...)
    return Base.afoldl(lf, lf(lf(a, b), c), xs...)
end
# Flatten AddBroadcasted + anything.
function (::typeof(LinearFunc(+)))(a::AddBroadcasted, b)
    return AddBroadcasted(addends(a)..., b)
end
function (::typeof(LinearFunc(+)))(a, b::AddBroadcasted)
    return AddBroadcasted(a, addends(b)...)
end
function (::typeof(LinearFunc(+)))(a::AddBroadcasted, b::AddBroadcasted)
    return AddBroadcasted(addends(a)..., addends(b)...)
end
(::typeof(LinearFunc(+)))(a) = a

# Subtraction.
function (::typeof(LinearFunc(-)))(a, b)
    return LinearFunc(+)(a, LinearFunc(*)(- 1, b))
end
(::typeof(LinearFunc(-)))(a) = LinearFunc(*)(-1, a)

# Division / left-division by scalars.
function (::typeof(LinearFunc(/)))(a, b::Number)
    return LinearFunc(*)(inv(b), a)
end
function (::typeof(LinearFunc(\)))(a::Number, b)
    return LinearFunc(*)(inv(a), b)
end

# Identity.
(::typeof(LinearFunc(identity)))(a) = a

# Fix1/Fix2 wrappers for scalar multiplication/division.
function (lf::LinearFunc{<:Base.Fix1{typeof(*)}})(a)
    return LinearFunc(*)(lf.f.x, a)
end
function (lf::LinearFunc{<:Base.Fix2{typeof(*)}})(a)
    return LinearFunc(*)(a, lf.f.x)
end
function (lf::LinearFunc{<:Base.Fix2{typeof(/)}})(a)
    return LinearFunc(/)(a, lf.f.x)
end

# Scaling of AddBroadcasted distributes.
function (::typeof(LinearFunc(*)))(α::Number, a::AddBroadcasted)
    return LinearFunc(+)(map(x -> LinearFunc(*)(α, x), addends(a))...)
end

# Conjugation of AddBroadcasted distributes.
function (::typeof(LinearFunc(conj)))(a::AddBroadcasted)
    return LinearFunc(+)(map(x -> LinearFunc(conj)(x), addends(a))...)
end

# Conjugation of Mul distributes.
function (::typeof(LinearFunc(conj)))(a::Mul)
    f = factors(a)
    return Mul(LinearFunc(conj)(f[1]), LinearFunc(conj)(f[2]))
end

# Scaling of Mul: wrap in ScaledBroadcasted.
function (::typeof(LinearFunc(*)))(α::Number, a::Mul)
    return ScaledBroadcasted(α, a)
end

# Number * Number passthrough (for broadcast lowering).
(::typeof(LinearFunc(*)))(a::Number, b::Number) = a * b

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
is_linear(x) = true
function is_linear(bc::BC.Broadcasted)
    return broadcast_is_linear(bc.f, bc.args...) && all(is_linear, bc.args)
end

to_linear(x) = x
function to_linear(bc::BC.Broadcasted)
    return LinearFunc(bc.f)(to_linear.(bc.args)...)
end

function broadcast_error(style, f)
    return throw(
        ArgumentError(
            "Only linear broadcast operations are supported for `$style`, got `$f`."
        )
    )
end
function broadcasted_linear(style::BC.BroadcastStyle, f, args...)
    bc = BC.Broadcasted(style, f, args)
    is_linear(bc) || broadcast_error(style, f)
    return to_linear(bc)
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
    return LinearFunc(+)(a, b)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle,
        ::typeof(+),
        a::AbstractArray,
        b::BC.Broadcasted
    )
    is_linear(b) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return LinearFunc(+)(a, to_linear(b))
end
function BC.broadcasted(
        ::LinearBroadcastedStyle,
        ::typeof(+),
        a::BC.Broadcasted,
        b::AbstractArray
    )
    is_linear(a) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return LinearFunc(+)(to_linear(a), b)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle, ::typeof(+), a::BC.Broadcasted, b::BC.Broadcasted
    )
    return error("Not implemented")
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), α::Number, a::AbstractArray)
    return LinearFunc(*)(α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), a::AbstractArray, α::Number)
    return LinearFunc(*)(a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(\), α::Number, a::AbstractArray)
    return LinearFunc(\)(α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(/), a::AbstractArray, α::Number)
    return LinearFunc(/)(a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(-), a::AbstractArray)
    return LinearFunc(-)(a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(conj), a::AbstractArray)
    return LinearFunc(conj)(a)
end
