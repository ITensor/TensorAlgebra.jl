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

# Generic axes(a, d) for LinearBroadcasted subtypes.
Base.axes(a::LinearBroadcasted, d::Int) = axes(a)[d]

# --- ScaledBroadcasted --------------------------------------------------------

struct ScaledBroadcasted{T, N, C <: Number, A} <: LinearBroadcasted
    coeff::C
    parent::A
    function ScaledBroadcasted(coeff::Number, a)
        T = Base.promote_op(*, typeof(coeff), eltype(a))
        return new{T, ndims(a), typeof(coeff), typeof(a)}(coeff, a)
    end
end

unscaled(a::ScaledBroadcasted) = a.parent
coeff(a::ScaledBroadcasted) = a.coeff

Base.axes(a::ScaledBroadcasted) = axes(unscaled(a))
Base.eltype(::Type{<:ScaledBroadcasted{T}}) where {T} = T
Base.eltype(::ScaledBroadcasted{T}) where {T} = T
Base.ndims(::Type{<:ScaledBroadcasted{<:Any, N}}) where {N} = N
Base.ndims(::ScaledBroadcasted{<:Any, N}) where {N} = N

function Base.similar(a::ScaledBroadcasted)
    return similar(unscaled(a), eltype(a), axes(a))
end
function Base.similar(a::ScaledBroadcasted, elt::Type)
    return similar(unscaled(a), elt, axes(a))
end

function Base.show(io::IO, a::ScaledBroadcasted)
    print(io, "*(", coeff(a), ", ", unscaled(a), ")")
    return nothing
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

iscall(::ScaledBroadcasted) = true
operation(::ScaledBroadcasted) = *
arguments(a::ScaledBroadcasted) = (coeff(a), unscaled(a))

# --- ConjBroadcasted ----------------------------------------------------------

struct ConjBroadcasted{T, N, A} <: LinearBroadcasted
    parent::A
    function ConjBroadcasted(a)
        return new{eltype(a), ndims(a), typeof(a)}(a)
    end
end

unconj(a::ConjBroadcasted) = a.parent

Base.axes(a::ConjBroadcasted) = axes(unconj(a))
Base.eltype(::Type{<:ConjBroadcasted{T}}) where {T} = T
Base.eltype(::ConjBroadcasted{T}) where {T} = T
Base.ndims(::Type{<:ConjBroadcasted{<:Any, N}}) where {N} = N
Base.ndims(::ConjBroadcasted{<:Any, N}) where {N} = N

function Base.similar(a::ConjBroadcasted)
    return similar(unconj(a), eltype(a), axes(a))
end
function Base.similar(a::ConjBroadcasted, elt::Type)
    return similar(unconj(a), elt, axes(a))
end

function Base.show(io::IO, a::ConjBroadcasted)
    print(io, "conj(", unconj(a), ")")
    return nothing
end

Base.conj(a::ConjBroadcasted) = unconj(a)
Base.adjoint(a::ConjBroadcasted) = transpose(unconj(a))
Base.transpose(a::ConjBroadcasted) = adjoint(unconj(a))

function FI.permuteddims(a::ConjBroadcasted, perm)
    return ConjBroadcasted(FI.permuteddims(unconj(a), perm))
end

SV.isstrided(a::ConjBroadcasted) = SV.isstrided(unconj(a))
SV.StridedView(a::ConjBroadcasted) = conj(SV.StridedView(unconj(a)))

iscall(::ConjBroadcasted) = true
operation(::ConjBroadcasted) = conj
arguments(a::ConjBroadcasted) = (unconj(a),)

# --- AddBroadcasted -----------------------------------------------------------

struct AddBroadcasted{T, N, Args <: Tuple} <: LinearBroadcasted
    args::Args
    function AddBroadcasted(args...)
        T = Base.promote_op(+, eltype.(args)...)
        N = if allequal(ndims, args)
            ndims(first(args))
        else
            error("All addends must have the same number of dimensions.")
        end
        return new{T, N, typeof(args)}(args)
    end
end

addends(a::AddBroadcasted) = a.args

Base.axes(a::AddBroadcasted) = BC.combine_axes(addends(a)...)
Base.eltype(::Type{<:AddBroadcasted{T}}) where {T} = T
Base.eltype(::AddBroadcasted{T}) where {T} = T
Base.ndims(::Type{<:AddBroadcasted{<:Any, N}}) where {N} = N
Base.ndims(::AddBroadcasted{<:Any, N}) where {N} = N

function Base.similar(a::AddBroadcasted)
    return similar(BC.Broadcasted(+, addends(a)), eltype(a))
end
function Base.similar(a::AddBroadcasted, elt::Type)
    return similar(BC.Broadcasted(+, addends(a)), elt)
end

function Base.show(io::IO, a::AddBroadcasted)
    print(io, "+(", join(addends(a), ", "), ")")
    return nothing
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

iscall(::AddBroadcasted) = true
operation(::AddBroadcasted) = +
arguments(a::AddBroadcasted) = addends(a)

# ---------------------------------------------------------------------------- #
# Mul — lazy matrix multiplication (standalone, not LinearBroadcasted)
# ---------------------------------------------------------------------------- #

# Same as `LinearAlgebra.matprod`, but duplicated here since it is private.
matprod(x, y) = x * y + x * y

struct Mul{T, N, A, B}
    a::A
    b::B
    function Mul(a, b)
        T = Base.promote_op(matprod, eltype(a), eltype(b))
        N = ndims(b)
        return new{T, N, typeof(a), typeof(b)}(a, b)
    end
end

factors(a::Mul) = (a.a, a.b)

Base.axes(a::Mul) = (axes(a.a, 1), axes(a.b, ndims(a.b)))
Base.axes(a::Mul, d::Int) = axes(a)[d]
Base.eltype(::Type{<:Mul{T}}) where {T} = T
Base.eltype(::Mul{T}) where {T} = T
Base.ndims(::Type{<:Mul{<:Any, N}}) where {N} = N
Base.ndims(::Mul{<:Any, N}) where {N} = N
Base.size(a::Mul) = length.(axes(a))

function Base.similar(a::Mul)
    return similar(BC.materialize(last(factors(a))), eltype(a), axes(a))
end
function Base.similar(a::Mul, elt::Type)
    return similar(BC.materialize(last(factors(a))), elt, axes(a))
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

function FI.permuteddims(a::Mul{<:Any, 2}, perm)
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
function Base.copyto!(dest::AbstractArray, src::ScaledBroadcasted)
    return add!(dest, src, true, false)
end
function Base.copyto!(dest::AbstractArray, src::ConjBroadcasted)
    return add!(dest, src, true, false)
end
function Base.copyto!(dest::AbstractArray, src::AddBroadcasted)
    return add!(dest, src, true, false)
end

# copyto! for Mul dispatches to mul!. Materialize factors first since
# they may be LinearBroadcasted types.
function Base.copyto!(dest::AbstractArray, src::Mul)
    return LA.mul!(dest, BC.materialize.(factors(src))..., true, false)
end

# add! for LinearBroadcasted subtypes.
function add!(dest::AbstractArray, src::ScaledBroadcasted, α::Number, β::Number)
    return add!(dest, unscaled(src), coeff(src) * α, β)
end

function add!(dest::AbstractArray, src::ConjBroadcasted, α::Number, β::Number)
    return add!(dest, unconj(src), α, β, Val(:conj))
end

# Default conj add! falls back to materializing conj.
function add!(dest::AbstractArray, src::AbstractArray, α::Number, β::Number, ::Val{:conj})
    return add!(dest, conj(src), α, β)
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

# Scaling: Number * AbstractArray
function (::LinearBroadcastFunction{typeof(*)})(α::Number, a::AbstractArray)
    return ScaledBroadcasted(α, a)
end
function (::LinearBroadcastFunction{typeof(*)})(a::AbstractArray, α::Number)
    return ScaledBroadcasted(α, a)
end
# Scaling of ScaledBroadcasted: absorb coefficient.
function (::LinearBroadcastFunction{typeof(*)})(α::Number, a::ScaledBroadcasted)
    return ScaledBroadcasted(α * coeff(a), unscaled(a))
end

# Conjugation.
function (::LinearBroadcastFunction{typeof(conj)})(a::AbstractArray)
    return ConjBroadcasted(a)
end
(::LinearBroadcastFunction{typeof(conj)})(a::AbstractArray{<:Real}) = a
(::LinearBroadcastFunction{typeof(conj)})(a::ConjBroadcasted) = unconj(a)
function (::LinearBroadcastFunction{typeof(conj)})(a::ScaledBroadcasted)
    return ScaledBroadcasted(
        conj(coeff(a)), LinearBroadcastFunction(conj)(unscaled(a))
    )
end

# Addition.
function (lf::LinearBroadcastFunction{typeof(+)})(a, b)
    return AddBroadcasted(a, b)
end
function (lf::LinearBroadcastFunction{typeof(+)})(a, b, c, xs...)
    return Base.afoldl(lf, lf(lf(a, b), c), xs...)
end
# Flatten AddBroadcasted + anything.
function (::LinearBroadcastFunction{typeof(+)})(a::AddBroadcasted, b)
    return AddBroadcasted(addends(a)..., b)
end
function (::LinearBroadcastFunction{typeof(+)})(a, b::AddBroadcasted)
    return AddBroadcasted(a, addends(b)...)
end
function (::LinearBroadcastFunction{typeof(+)})(a::AddBroadcasted, b::AddBroadcasted)
    return AddBroadcasted(addends(a)..., addends(b)...)
end
(::LinearBroadcastFunction{typeof(+)})(a) = a

# Subtraction.
function (::LinearBroadcastFunction{typeof(-)})(a, b)
    return LinearBroadcastFunction(+)(a, LinearBroadcastFunction(*)(- 1, b))
end
(::LinearBroadcastFunction{typeof(-)})(a) = LinearBroadcastFunction(*)(-1, a)

# Division / left-division by scalars.
function (::LinearBroadcastFunction{typeof(/)})(a, b::Number)
    return LinearBroadcastFunction(*)(inv(b), a)
end
function (::LinearBroadcastFunction{typeof(\)})(a::Number, b)
    return LinearBroadcastFunction(*)(inv(a), b)
end

# Identity.
(::LinearBroadcastFunction{typeof(identity)})(a) = a

# Fix1/Fix2 wrappers for scalar multiplication/division.
function (lf::LinearBroadcastFunction{<:Base.Fix1{typeof(*)}})(a)
    return LinearBroadcastFunction(*)(lf.f.x, a)
end
function (lf::LinearBroadcastFunction{<:Base.Fix2{typeof(*)}})(a)
    return LinearBroadcastFunction(*)(a, lf.f.x)
end
function (lf::LinearBroadcastFunction{<:Base.Fix2{typeof(/)}})(a)
    return LinearBroadcastFunction(/)(a, lf.f.x)
end

# Scaling of AddBroadcasted distributes.
function (::LinearBroadcastFunction{typeof(*)})(α::Number, a::AddBroadcasted)
    return LinearBroadcastFunction(+)(
        map(x -> LinearBroadcastFunction(*)(α, x), addends(a))...
    )
end

# Conjugation of AddBroadcasted distributes.
function (::LinearBroadcastFunction{typeof(conj)})(a::AddBroadcasted)
    return LinearBroadcastFunction(+)(
        map(x -> LinearBroadcastFunction(conj)(x), addends(a))...
    )
end

# Conjugation of Mul distributes.
function (::LinearBroadcastFunction{typeof(conj)})(a::Mul)
    f = factors(a)
    return Mul(LinearBroadcastFunction(conj)(f[1]), LinearBroadcastFunction(conj)(f[2]))
end

# Scaling of Mul: wrap in ScaledBroadcasted.
function (::LinearBroadcastFunction{typeof(*)})(α::Number, a::Mul)
    return ScaledBroadcasted(α, a)
end

# Number * Number passthrough (for broadcast lowering).
(::LinearBroadcastFunction{typeof(*)})(a::Number, b::Number) = a * b

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
    return LinearBroadcastFunction(bc.f)(to_linear.(bc.args)...)
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
struct LinearBroadcastedStyle{N, Style <: BC.AbstractArrayStyle{N}} <: BC.AbstractArrayStyle{N}
    style::Style
end
# TODO: This empty constructor is required in some Julia versions below v1.12 (such as
# Julia v1.10), try deleting it once we drop support for those versions.
function LinearBroadcastedStyle{N, Style}() where {N, Style <: BC.AbstractArrayStyle{N}}
    return LinearBroadcastedStyle{N, Style}(Style())
end
function LinearBroadcastedStyle{N, Style}(::Val{M}) where {M, N, Style <: BC.AbstractArrayStyle{N}}
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
function BC.BroadcastStyle(::Type{<:ScaledBroadcasted{<:Any, <:Any, <:Any, A}}) where {A}
    return LinearBroadcastedStyle(BC.BroadcastStyle(A))
end
function BC.BroadcastStyle(::Type{<:ConjBroadcasted{<:Any, <:Any, A}}) where {A}
    return LinearBroadcastedStyle(BC.BroadcastStyle(A))
end
function BC.BroadcastStyle(::Type{<:AddBroadcasted{<:Any, <:Any, Args}}) where {Args}
    style = Base.promote_op(BC.combine_styles, fieldtypes(Args)...)()
    return LinearBroadcastedStyle(style)
end
function BC.BroadcastStyle(::Type{<:Mul{<:Any, <:Any, A, B}}) where {A, B}
    style = BC.BroadcastStyle(BC.BroadcastStyle(A), BC.BroadcastStyle(B))
    return LinearBroadcastedStyle(style)
end

# Broadcast.materialize for LinearBroadcasted and Mul.
BC.materialize(a::LinearBroadcasted) = copy(a)
BC.materialize(a::Mul) = copy(a)

# Backup definition: for broadcast operations that don't preserve lazy types
# (such as nonlinear operations), convert back to Broadcasted expressions.
BC.broadcasted(::LinearBroadcastedStyle, f, args...) = BC.Broadcasted(f, to_broadcasted.(args))

# Linear broadcast operations produce LinearBroadcasted / Mul types.
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(+), a::AbstractArray, b::AbstractArray)
    return LinearBroadcastFunction(+)(a, b)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(+), a::AbstractArray, b::BC.Broadcasted)
    is_linear(b) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return LinearBroadcastFunction(+)(a, to_linear(b))
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(+), a::BC.Broadcasted, b::AbstractArray)
    is_linear(a) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return LinearBroadcastFunction(+)(to_linear(a), b)
end
function BC.broadcasted(
        ::LinearBroadcastedStyle, ::typeof(+), a::BC.Broadcasted, b::BC.Broadcasted
    )
    return error("Not implemented")
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), α::Number, a::AbstractArray)
    return LinearBroadcastFunction(*)(α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(*), a::AbstractArray, α::Number)
    return LinearBroadcastFunction(*)(a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(\), α::Number, a::AbstractArray)
    return LinearBroadcastFunction(\)(α, a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(/), a::AbstractArray, α::Number)
    return LinearBroadcastFunction(/)(a, α)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(-), a::AbstractArray)
    return LinearBroadcastFunction(-)(a)
end
function BC.broadcasted(::LinearBroadcastedStyle, ::typeof(conj), a::AbstractArray)
    return LinearBroadcastFunction(conj)(a)
end
