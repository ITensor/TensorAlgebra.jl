using Base.Broadcast: Broadcast as BC
using LinearAlgebra: LinearAlgebra as LA

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
"""
abstract type LinearBroadcasted end

# Generic interface for LinearBroadcasted subtypes.
Base.axes(a::LinearBroadcasted, d::Int) = axes(a)[d]
Base.similar(a::LinearBroadcasted) = similar(a, eltype(a))
# Forward to 2-arg broadcast `similar` so the result preserves the operands' backend type.
Base.similar(a::LinearBroadcasted, elt::Type) = similar(BC.Broadcasted(a), elt)
function Base.show(io::IO, a::LinearBroadcasted)
    print(io, operation(a), "(", join(arguments(a), ", "), ")")
    return nothing
end
iscall(::LinearBroadcasted) = true

# Convert LinearBroadcasted back to Broadcasted (inverse of tryflattenlinear).
# Uses BC.Broadcasted constructor directly (not BC.broadcasted) to avoid style-based
# dispatch that could re-enter LinearBroadcasted conversion.
function BC.Broadcasted(a::LinearBroadcasted)
    args = map(arguments(a)) do arg
        return arg isa LinearBroadcasted ? BC.Broadcasted(arg) : arg
    end
    return BC.Broadcasted(BC.combine_styles(args...), operation(a), args)
end

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

operation(::ScaledBroadcasted) = *
arguments(a::ScaledBroadcasted) = (coeff(a), unscaled(a))

# --- ConjBroadcasted ----------------------------------------------------------

"""
    ConjBroadcasted(parent)

Lazy conjugate of `parent` in a [`LinearBroadcasted`](@ref) expression, produced when a broadcast
applies `conj`. Its axes are the conjugated axes of `parent`. Unlike [`ConjArray`](@ref) it is not
an `AbstractArray`, so it can also wrap a non-array operand such as a `TensorMap`.
"""
struct ConjBroadcasted{P} <: LinearBroadcasted
    parent::P
end

Base.parent(a::ConjBroadcasted) = a.parent

Base.axes(a::ConjBroadcasted) = map(conj, axes(parent(a)))
Base.eltype(a::ConjBroadcasted) = eltype(parent(a))
Base.ndims(a::ConjBroadcasted) = ndims(parent(a))

operation(::ConjBroadcasted) = conj
arguments(a::ConjBroadcasted) = (parent(a),)

function BC.BroadcastStyle(::Type{<:ConjBroadcasted{P}}) where {P}
    return BC.BroadcastStyle(P)
end

# --- AddBroadcasted -----------------------------------------------------------

struct AddBroadcasted{Args <: Tuple} <: LinearBroadcasted
    args::Args
    AddBroadcasted(args...) = new{typeof(args)}(args)
end

addends(a::AddBroadcasted) = a.args

# A linear combination does not broadcast differing shapes. Checked through TensorAlgebra's `axes`
# rather than Base's `combine_axes`, which would call `Base.axes` on a non-`AbstractArray` operand.
function Base.axes(a::AddBroadcasted)
    axs = map(axes, addends(a))
    ax = first(axs)
    all(==(ax), axs) ||
        throw(DimensionMismatch("linear-combination operands have mismatched axes: $axs"))
    return ax
end
Base.eltype(a::AddBroadcasted) = Base.promote_op(+, eltype.(addends(a))...)
Base.ndims(a::AddBroadcasted) = ndims(first(addends(a)))

operation(::AddBroadcasted) = +
arguments(a::AddBroadcasted) = addends(a)

# ---------------------------------------------------------------------------- #
# Mul — lazy matrix multiplication (standalone, not LinearBroadcasted)
# ---------------------------------------------------------------------------- #

# The element type of a matrix-product accumulation, modeled on `LinearAlgebra.matprod` (private
# there, so inlined): a sum of products, so `promote_op(matprod, ...)` widens exactly as an
# accumulating inner loop would.
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
# Materialization protocol: copy, copyto!, scaleadd!
# ---------------------------------------------------------------------------- #

function Base.copy(a::LinearBroadcasted)
    return copyto!(similar(a), a)
end

function Base.copy(a::Mul)
    return copyto!(similar(a), a)
end

# Kept `AbstractArray`-bound, since widening `dest` to `Any` collides with Base's `copyto!` methods.
function Base.copyto!(dest::AbstractArray, src::LinearBroadcasted)
    return scaleadd!(dest, src, true, false)
end

# The factors may be `LinearBroadcasted`, so materialize them first.
function Base.copyto!(dest::AbstractArray, src::Mul)
    return LA.mul!(dest, BC.materialize.(factors(src))...)
end

function _bipermutedimsconjadd!(
        algorithm, dest, isconj::Bool, src::ConjBroadcasted,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    return bipermutedimsconjadd!(
        algorithm, dest, !isconj, parent(src), perm_codomain, perm_domain, α, β
    )
end
function _bipermutedimsconjadd!(
        algorithm, dest, isconj::Bool, src::ScaledBroadcasted,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    c = isconj ? conj(coeff(src)) : coeff(src)
    return bipermutedimsconjadd!(
        algorithm, dest, isconj, unscaled(src), perm_codomain, perm_domain, c * α, β
    )
end
function _bipermutedimsconjadd!(
        algorithm, dest, isconj::Bool, src::AddBroadcasted,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    args = addends(src)
    bipermutedimsconjadd!(
        algorithm, dest, isconj, first(args), perm_codomain, perm_domain, α, β
    )
    for a in Base.tail(args)
        bipermutedimsconjadd!(
            algorithm, dest, isconj, a, perm_codomain, perm_domain, α, true
        )
    end
    return dest
end
function _bipermutedimsconjadd!(
        algorithm, dest, isconj::Bool, src::Mul,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    # TODO: Replace with contractconjadd! to avoid materializing the Mul intermediate.
    return bipermutedimsconjadd!(
        algorithm, dest, isconj, copy(src), perm_codomain, perm_domain, α, β
    )
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
linearbroadcasted(::typeof(*), α::Number, a) = ScaledBroadcasted(α, a)
linearbroadcasted(::typeof(*), a, α::Number) = ScaledBroadcasted(α, a)
# Scaling of ScaledBroadcasted: absorb coefficient.
function linearbroadcasted(::typeof(*), α::Number, a::ScaledBroadcasted)
    return ScaledBroadcasted(α * coeff(a), unscaled(a))
end
# Conjugation. A nested conjugate cancels.
linearbroadcasted(::typeof(conj), a) = ConjBroadcasted(a)
linearbroadcasted(::typeof(conj), a::ConjBroadcasted) = parent(a)
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

Whether the broadcast node `(f, args...)` can be expressed as a `LinearBroadcasted`.
Packages can add methods for additional linear operations.
"""
# A non-`Number` leaf is treated as array-like, which separates scaling (linear) from a scalar
# shift like `a .+ 1` (affine) or an elementwise array product (nonlinear). A leaf that is neither
# array-like nor a `Number` is assumed linear here and errors later when it is materialized.
islinearbroadcast(f, args...) = false

islinearbroadcast(::typeof(identity), ::Any) = true
islinearbroadcast(::typeof(identity), ::Number) = false

islinearbroadcast(::typeof(+), ::Any...) = true
islinearbroadcast(::typeof(+), ::Number) = false
islinearbroadcast(::typeof(+), ::Number, ::Any) = false
islinearbroadcast(::typeof(+), ::Any, ::Number) = false
islinearbroadcast(::typeof(+), ::Number, ::Number) = false

islinearbroadcast(::typeof(-), ::Any) = true
islinearbroadcast(::typeof(-), ::Number) = false
islinearbroadcast(::typeof(-), ::Any, ::Any) = true
islinearbroadcast(::typeof(-), ::Number, ::Any) = false
islinearbroadcast(::typeof(-), ::Any, ::Number) = false
islinearbroadcast(::typeof(-), ::Number, ::Number) = false

islinearbroadcast(::typeof(*), ::Number, ::Any) = true
islinearbroadcast(::typeof(*), ::Any, ::Number) = true
islinearbroadcast(::typeof(*), ::Any, ::Any) = false
islinearbroadcast(::typeof(*), ::Number, ::Number) = true

islinearbroadcast(::typeof(\), ::Number, ::Any) = true
islinearbroadcast(::typeof(/), ::Any, ::Number) = true

islinearbroadcast(::typeof(conj), ::Any) = true
islinearbroadcast(::typeof(conj), ::Number) = false

islinearbroadcast(::Base.Fix1{typeof(*), <:Number}, ::Any) = true
islinearbroadcast(::Base.Fix2{typeof(*), <:Number}, ::Any) = true
islinearbroadcast(::Base.Fix2{typeof(/), <:Number}, ::Any) = true

"""
    tryflattenlinear(bc::Broadcasted) -> LinearBroadcasted or nothing

Recursively convert a `Broadcasted` tree to a `LinearBroadcasted` tree.
Returns `nothing` if any node is not linear (as determined by `islinearbroadcast`).

Broadcast styles call this from `Base.copy(::Broadcasted{MyStyle})` to
opt into linear broadcasting at materialization time.
"""
tryflattenlinear(x) = x
function tryflattenlinear(bc::BC.Broadcasted)
    islinearbroadcast(bc.f, bc.args...) || return nothing
    args = map(tryflattenlinear, bc.args)
    any(isnothing, args) && return nothing
    return linearbroadcasted(bc.f, args...)
end

"""
    flattenlinear(bc::Broadcasted) -> LinearBroadcasted

Like [`tryflattenlinear`](@ref), but throw an `ArgumentError` when the expression is not
linear instead of returning `nothing`.
"""
function flattenlinear(bc)
    lb = tryflattenlinear(bc)
    isnothing(lb) && throw(ArgumentError("broadcast expression is not linear"))
    return lb
end

# BroadcastStyle for LinearBroadcasted subtypes — delegate to the wrapped array type.
function BC.BroadcastStyle(::Type{<:ScaledBroadcasted{<:Any, A}}) where {A}
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
