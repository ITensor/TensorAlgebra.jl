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

# Convert LinearBroadcasted back to Broadcasted (inverse of tryflattenlinear).
# Uses BC.Broadcasted constructor directly (not BC.broadcasted) to avoid style-based
# dispatch that could re-enter LinearBroadcasted conversion.
function BC.Broadcasted(a::LinearBroadcasted)
    args = map(arguments(a)) do arg
        return arg isa LinearBroadcasted ? BC.Broadcasted(arg) : arg
    end
    return BC.Broadcasted(BC.combine_styles(args...), operation(a), args)
end

function Base.similar(a::LinearBroadcasted, elt::Type, ax)
    return similar(BC.Broadcasted(a), elt, ax)
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

Lazy conjugate node in the linear-combination broadcast fold, the [`LinearBroadcasted`](@ref)
counterpart of `ScaledBroadcasted`/`AddBroadcasted`. Holds `parent` (any array-like operand or
backend tensor, not necessarily an `AbstractArray`) and presents its axes conjugated (dualized);
the op primitives absorb it by unwrapping to the parent and folding `conj` into their `op` (see
the `bipermutedimsopadd!` method below). Produced internally by the `conj` lowering of a
broadcast (`linearbroadcasted(conj, a)`), not a user-facing lazy-conjugate wrapper. Because it
is not an `AbstractArray`, it works for non-array backends (e.g. a `TensorMap`) as well as dense
arrays.
"""
struct ConjBroadcasted{P} <: LinearBroadcasted
    parent::P
end

Base.parent(a::ConjBroadcasted) = a.parent

# Conjugating an axis is identity for plain integer ranges and dualizes graded axes (where
# `conj` is overloaded as `dual` downstream).
Base.axes(a::ConjBroadcasted) = map(conj, axes(parent(a)))
Base.eltype(a::ConjBroadcasted) = eltype(parent(a))
Base.ndims(a::ConjBroadcasted) = ndims(parent(a))

operation(::ConjBroadcasted) = conj
arguments(a::ConjBroadcasted) = (parent(a),)

function BC.BroadcastStyle(::Type{<:ConjBroadcasted{P}}) where {P}
    return BC.BroadcastStyle(P)
end

# Absorb a `ConjBroadcasted` source by unwrapping to the parent and folding `conj` into `op`
# (`identity -> conj`, `conj -> identity`). Mirrors the `PermutedDims` absorption, which
# instead folds its permutation into `perm`. `dest` is unrestricted so a non-`AbstractArray`
# backend (e.g. a `TensorMap`) receives it and applies `conj` through its own op-aware
# `bipermutedimsopadd!`.
function bipermutedimsopadd!(
        dest, op, src::ConjBroadcasted,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    return bipermutedimsopadd!(
        dest, _compose_op(op, conj), parent(src),
        perm_codomain, perm_domain,
        α, β
    )
end

# --- AddBroadcasted -----------------------------------------------------------

struct AddBroadcasted{Args <: Tuple} <: LinearBroadcasted
    args::Args
    AddBroadcasted(args...) = new{typeof(args)}(args)
end

addends(a::AddBroadcasted) = a.args

# All addends of a linear combination share axes (a linear combination does not broadcast
# differing shapes), so combine by verifying equality through `axes` (TensorAlgebra's, which
# works for a non-`AbstractArray` backend like a `TensorMap`) rather than Base's `combine_axes`,
# which would call `Base.axes`/`Base.size` on the operands. A mismatch (e.g. a half-conjugated
# `conj.(a) .- b`, whose dualized and non-dualized axes differ) throws here.
function Base.axes(a::AddBroadcasted)
    axs = map(axes, addends(a))
    ax = first(axs)
    all(x -> x == ax, axs) ||
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
# Stays `AbstractArray`-bound: these overload `Base.copyto!`, so widening `dest` to `Any`
# collides with Base's own methods. A non-array destination (e.g. a wrapped `TensorMap`)
# gets an `AbstractTensorMap`-specific `copyto!` from the backend extension instead.
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
        dest, op, src::ScaledBroadcasted, perm, α::Number, β::Number
    )
    return permutedimsopadd!(dest, op, unscaled(src), perm, op(coeff(src)) * α, β)
end

function permutedimsopadd!(
        dest, op, src::AddBroadcasted, perm, α::Number, β::Number
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
        dest, op, src::Mul, perm, α::Number, β::Number
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
linearbroadcasted(::typeof(*), α::Number, a) = ScaledBroadcasted(α, a)
linearbroadcasted(::typeof(*), a, α::Number) = ScaledBroadcasted(α, a)
# Scaling of ScaledBroadcasted: absorb coefficient.
function linearbroadcasted(::typeof(*), α::Number, a::ScaledBroadcasted)
    return ScaledBroadcasted(α * coeff(a), unscaled(a))
end
# Conjugation lowers to the `ConjBroadcasted` node; a nested conjugate cancels (`conj∘conj =
# identity`). A scaled conjugate (e.g. `conj.(a) ./ β`) is handled by the generic scaling method
# above, whose operand slot is untyped, so it wraps the `ConjBroadcasted` in a `ScaledBroadcasted`.
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

Per-node trait: can `(f, args...)` be expressed as a `LinearBroadcasted`?
Extensible by downstream packages for additional linear operations.
"""
# A leaf operand is treated as an array/tensor by default (`::Any`), and `::Number` marks the
# scalars. That distinction is what makes the predicate correct: scaling by a scalar is linear
# while a scalar shift (`a .+ 1`) is affine, and an elementwise array product is nonlinear while
# scaling is not. Because the array-like slot is `::Any`, any leaf type participates with no
# change here — it need not be an `AbstractArray` (e.g. a `PermutedDims` wrapper, or a backend
# tensor); the fold absorbs it via `bipermutedimsopadd!`. An operand that is neither a genuine
# array-like leaf nor a `Number` (say a scalar buried in an n-ary `+`) is assumed linear here and
# errors later at the fold rather than silently producing a wrong result.
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

"""
    flattenlinear(bc::Broadcasted) -> LinearBroadcasted

Like [`tryflattenlinear`](@ref), but throw an `ArgumentError` when the expression is not
linear instead of returning `nothing`. The erroring counterpart to `tryflattenlinear`,
following the `parse`/`tryparse` convention.
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
