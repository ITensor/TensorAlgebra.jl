import Base.Broadcast as BC
import LinearAlgebra as LA
import StridedViews as SV

# TermInterface-like interface.
iscall(x) = false
function operation end
function arguments end

# Generic logic for lazy array linear algebra operations.
function +ₗ(a::AbstractArray, b::AbstractArray, c::AbstractArray, xs::AbstractArray...)
    return Base.afoldl(+ₗ, +ₗ(+ₗ(a, b), c), xs...)
end
-ₗ(a::AbstractArray, b::AbstractArray) = a +ₗ (-b)
function *ₗ(a::AbstractArray, b::AbstractArray, c::AbstractArray, xs::AbstractArray...)
    return Base.afoldl(*ₗ, *ₗ(*ₗ(a, b), c), xs...)
end
*ₗ(a::AbstractArray, b::Number) = b *ₗ a
\ₗ(a::Number, b::AbstractArray) = inv(a) *ₗ b
/ₗ(a::AbstractArray, b::Number) = a *ₗ inv(b)
+ₗ(a::AbstractArray) = a
-ₗ(a::AbstractArray) = -1 *ₗ a
conjed(a::AbstractArray{<:Real}) = a

lazy_function(f) = error("No lazy function defined for `$f`.")
lazy_function(::typeof(+)) = +ₗ
lazy_function(::typeof(-)) = -ₗ
lazy_function(::typeof(*)) = *ₗ
lazy_function(::typeof(/)) = /ₗ
lazy_function(::typeof(\)) = \ₗ
lazy_function(::typeof(conj)) = conjed

broadcast_is_linear(f, args...) = false
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
is_linear(x) = true
function is_linear(bc::BC.Broadcasted)
    return broadcast_is_linear(bc.f, bc.args...) && all(is_linear, bc.args)
end

to_linear(x) = x
to_linear(bc::BC.Broadcasted) = lazy_function(bc.f)(to_linear.(bc.args)...)
# TODO: Use `Broadcast.broadcastable` interface for this?
to_broadcasted(x) = x
function to_broadcasted(a::AbstractArray)
    (BC.BroadcastStyle(typeof(a)) isa LazyArrayStyle) || return a
    return BC.broadcasted(operation(a), to_broadcasted.(arguments(a))...)
end
to_broadcasted(bc::BC.Broadcasted) = BC.Broadcasted(bc.f, to_broadcasted.(bc.args))

# For lazy arrays, define Broadcast methods in terms of lazy operations.
struct LazyArrayStyle{N, Style <: BC.AbstractArrayStyle{N}} <: BC.AbstractArrayStyle{N}
    style::Style
end
function LazyArrayStyle{N, Style}(::Val{M}) where {M, N, Style <: BC.AbstractArrayStyle{N}}
    return LazyArrayStyle(Style(Val(M)))
end
function BC.BroadcastStyle(style1::LazyArrayStyle, style2::LazyArrayStyle)
    style = BC.BroadcastStyle(style1.style, style2.style)
    style ≡ BC.Unknown() && return BC.Unknown()
    return LazyArrayStyle(style)
end
function Base.similar(bc::BC.Broadcasted{<:LazyArrayStyle}, elt::Type, ax)
    return similar(BC.Broadcasted(bc.style.style, bc.f, bc.args, bc.axes), elt, ax)
end
# Backup definition, for broadcast operations that don't preserve LazyArrays
# (such as nonlinear operations), convert back to Broadcasted expressions.
BC.broadcasted(::LazyArrayStyle, f, args...) = BC.Broadcasted(f, to_broadcasted.(args))
BC.broadcasted(::LazyArrayStyle, ::typeof(+), a::AbstractArray, b::AbstractArray) = a +ₗ b
function BC.broadcasted(::LazyArrayStyle, ::typeof(+), a::AbstractArray, b::BC.Broadcasted)
    is_linear(b) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return a +ₗ to_linear(b)
end
function BC.broadcasted(::LazyArrayStyle, ::typeof(+), a::BC.Broadcasted, b::AbstractArray)
    is_linear(a) || return BC.Broadcasted(+, to_broadcasted.((a, b)))
    return to_linear(a) +ₗ b
end
function BC.broadcasted(::LazyArrayStyle, ::typeof(+), a::BC.Broadcasted, b::BC.Broadcasted)
    return error("Not implemented")
end
BC.broadcasted(::LazyArrayStyle, ::typeof(*), α::Number, a::AbstractArray) = α *ₗ a
BC.broadcasted(::LazyArrayStyle, ::typeof(*), a::AbstractArray, α::Number) = a *ₗ α
BC.broadcasted(::LazyArrayStyle, ::typeof(\), α::Number, a::AbstractArray) = α \ₗ a
BC.broadcasted(::LazyArrayStyle, ::typeof(/), a::AbstractArray, α::Number) = a /ₗ α
BC.broadcasted(::LazyArrayStyle, ::typeof(-), a::AbstractArray) = -ₗ(a)
BC.broadcasted(::LazyArrayStyle, ::typeof(conj), a::AbstractArray) = conjed(a)

# Base overloads for lazy arrays.
function show_lazy(io::IO, a::AbstractArray)
    print(io, operation(a), "(", join(arguments(a), ", "), ")")
    return nothing
end
function show_lazy(io::IO, mime::MIME"text/plain", a::AbstractArray)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

# Generic constructors, accessors, and properties for ScaledArrays.
*ₗ(α::Number, a::AbstractArray) = ScaledArray(α, a)
unscaled(a::AbstractArray) = a
unscaled_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(unscaled, arrayt)
coeff(a::AbstractArray) = true
coeff_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(coeff, arrayt)
scaled_eltype(coeff::Number, a::AbstractArray) =
    Base.promote_op(*, typeof(coeff), eltype(a))

# Base overloads for ScaledArrays.
axes_scaled(a::AbstractArray) = axes(unscaled(a))
size_scaled(a::AbstractArray) = size(unscaled(a))
similar_scaled(a::AbstractArray) = similar(unscaled(a))
similar_scaled(a::AbstractArray, elt::Type) = similar(unscaled(a), elt)
similar_scaled(a::AbstractArray, ax) = similar(unscaled(a), ax)
similar_scaled(a::AbstractArray, elt::Type, ax) = similar(unscaled(a), elt, ax)
copyto!_scaled(dest::AbstractArray, src::AbstractArray) = add!(dest, src, true, false)
show_scaled(io::IO, a::AbstractArray) = show_lazy(io, a)
show_scaled(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for ScaledArrays.
materialize_scaled(a::AbstractArray) = copy(a)
BroadcastStyle_scaled(arrayt::Type{<:AbstractArray}) =
    LazyArrayStyle(BC.BroadcastStyle(unscaled_type(arrayt)))

# LinearAlgebra overloads for ScaledArrays.
mul!_scaled(dest::AbstractArray, a::AbstractArray, b::AbstractArray, α::Number, β::Number) =
    LA.mul!(dest, unscaled(a), unscaled(b), coeff(a) * coeff(b) * α, β)

# Lazy operations for ScaledArrays.
mulled_scaled(α::Number, a::AbstractArray) = (α * coeff(a)) *ₗ unscaled(a)
mulled_scaled(a::AbstractArray, b::AbstractArray) =
    (coeff(a) * coeff(b)) *ₗ (unscaled(a) *ₗ unscaled(b))
conjed_scaled(a::AbstractArray) = conj(coeff(a)) *ₗ conjed(unscaled(a))

# TensorAlgebra overloads for ScaledArrays.
add!_scaled(dest::AbstractArray, src::AbstractArray, α::Number, β::Number) =
    add!(dest, unscaled(src), coeff(src) * α, β)

# TermInterface-like overloads for ScaledArrays.
iscall_scaled(::AbstractArray) = true
operation_scaled(::AbstractArray) = *
arguments_scaled(a::AbstractArray) = (coeff(a), unscaled(a))

macro scaledarray_type(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $ScaledArray{T, N, P <: AbstractArray{<:Any, N}, C <: Number} <:
                $AbstractArray{T, N}
                coeff::C
                parent::P
                function $ScaledArray(coeff::Number, a::AbstractArray)
                    T = TensorAlgebra.scaled_eltype(coeff, a)
                    return new{T, ndims(a), typeof(a), typeof(coeff)}(coeff, a)
                end
            end
            TensorAlgebra.unscaled(a::$ScaledArray) = a.parent
            TensorAlgebra.unscaled_type(arrayt::Type{<:$ScaledArray}) =
                fieldtype(arrayt, :parent)
            TensorAlgebra.coeff(a::$ScaledArray) = a.coeff
            TensorAlgebra.coeff_type(arrayt::Type{<:$ScaledArray}) =
                fieldtype(arrayt, :coeff)
        end
    )
end

macro scaledarray_base(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.axes(a::$ScaledArray) =
                TensorAlgebra.axes_scaled(a)
            Base.size(a::$ScaledArray) =
                TensorAlgebra.size_scaled(a)
            Base.similar(a::$ScaledArray) =
                TensorAlgebra.similar_scaled(a)
            Base.similar(a::$ScaledArray, elt::Type) =
                TensorAlgebra.similar_scaled(a, elt)
            Base.similar(a::$ScaledArray, ax) =
                TensorAlgebra.similar_scaled(a, ax)
            Base.similar(a::$ScaledArray, elt::Type, ax) =
                TensorAlgebra.similar_scaled(a, elt, ax)
            Base.similar(a::$ScaledArray, elt::Type, ax::Dims) =
                TensorAlgebra.similar_scaled(a, elt, ax)
            Base.copyto!(dest::$AbstractArray, src::$ScaledArray) =
                TensorAlgebra.copyto!_scaled(dest, src)
            Base.show(io::IO, a::$ScaledArray) =
                TensorAlgebra.show_scaled(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$ScaledArray) =
                TensorAlgebra.show_scaled(io, mime, a)
        end
    )
end

macro scaledarray_broadcast(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$ScaledArray) =
                TensorAlgebra.materialize_scaled(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$ScaledArray}) =
                TensorAlgebra.BroadcastStyle_scaled(arrayt)
        end
    )
end

macro scaledarray_linearalgebra(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            function TensorAlgebra.LA.mul!(
                    dest::$AbstractArray{<:Any, 2},
                    a::$ScaledArray{<:Any, 2},
                    b::$ScaledArray{<:Any, 2},
                    α::Number, β::Number,
                )
                return TensorAlgebra.mul!_scaled(dest, a, b, α, β)
            end
            function TensorAlgebra.LA.mul!(
                    dest::$AbstractArray{<:Any, 2},
                    a::$AbstractArray{<:Any, 2},
                    b::$ScaledArray{<:Any, 2},
                    α::Number, β::Number,
                )
                return TensorAlgebra.mul!_scaled(dest, a, b, α, β)
            end
            function TensorAlgebra.LA.mul!(
                    dest::$AbstractArray{<:Any, 2},
                    a::$ScaledArray{<:Any, 2},
                    b::$AbstractArray{<:Any, 2},
                    α::Number, β::Number,
                )
                return TensorAlgebra.mul!_scaled(dest, a, b, α, β)
            end
        end
    )
end

macro scaledarray_lazy(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.:(*ₗ)(α::Number, a::$ScaledArray) =
                TensorAlgebra.mulled_scaled(α, a)
            TensorAlgebra.:(*ₗ)(a::$ScaledArray, b::$ScaledArray) =
                TensorAlgebra.mulled_scaled(a, b)
            TensorAlgebra.:(*ₗ)(a::$AbstractArray, b::$ScaledArray) =
                TensorAlgebra.mulled_scaled(a, b)
            TensorAlgebra.:(*ₗ)(a::$ScaledArray, b::$AbstractArray) =
                TensorAlgebra.mulled_scaled(a, b)
            TensorAlgebra.conjed(a::$ScaledArray) =
                TensorAlgebra.conjed_scaled(a)
        end
    )
end

macro scaledarray_tensoralgebra(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            function TensorAlgebra.add!(
                    dest::$AbstractArray, src::$ScaledArray, α::Number, β::Number
                )
                return TensorAlgebra.add!_scaled(dest, src, α, β)
            end
        end
    )
end

macro scaledarray_terminterface(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.iscall(a::$ScaledArray) = TensorAlgebra.iscall_scaled(a)
            TensorAlgebra.operation(a::$ScaledArray) = TensorAlgebra.operation_scaled(a)
            TensorAlgebra.arguments(a::$ScaledArray) = TensorAlgebra.arguments_scaled(a)
        end
    )
end

macro scaledarray(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.@scaledarray_base $ScaledArray $AbstractArray
            TensorAlgebra.@scaledarray_broadcast $ScaledArray $AbstractArray
            TensorAlgebra.@scaledarray_lazy $ScaledArray $AbstractArray
            TensorAlgebra.@scaledarray_linearalgebra $ScaledArray $AbstractArray
            TensorAlgebra.@scaledarray_tensoralgebra $ScaledArray $AbstractArray
            TensorAlgebra.@scaledarray_terminterface $ScaledArray $AbstractArray
        end
    )
end

# Generic constructors for ConjArrays.
conjed(a::AbstractArray) = ConjArray(a)
conjed_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(conj, arrayt)

# Base overloads for ConjArrays.
axes_conj(a::AbstractArray) = axes(conjed(a))
size_conj(a::AbstractArray) = size(conjed(a))
similar_conj(a::AbstractArray, elt::Type) = similar(conjed(a), elt)
similar_conj(a::AbstractArray, elt::Type, ax) = similar(conjed(a), elt, ax)
similar_conj(a::AbstractArray, ax) = similar(conjed(a), ax)
copyto!_conj(dest::AbstractArray, src::AbstractArray) = add!(dest, src, true, false)
show_conj(io::IO, a::AbstractArray) = show_lazy(io, a)
show_conj(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for ConjArrays.
materialize_conj(a::AbstractArray) = copy(a)
BroadcastStyle_conj(arrayt::Type{<:AbstractArray}) =
    LazyArrayStyle(BC.BroadcastStyle(conjed_type(arrayt)))

# StridedViews overloads for ConjArrays.
isstrided_conj(a::AbstractArray) = SV.isstrided(conjed(a))
StridedView_conj(a::AbstractArray) = conj(SV.StridedView(conjed(a)))

# TermInterface-like overloads for ConjArrays.
iscall_conj(::AbstractArray) = true
operation_conj(::AbstractArray) = conj
arguments_conj(a::AbstractArray) = (conjed(a),)

macro conjarray_type(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $ConjArray{T, N, P <: AbstractArray{T, N}} <: $AbstractArray{T, N}
                parent::P
            end
            TensorAlgebra.conjed(a::$ConjArray) = a.parent
        end
    )
end

macro conjarray_base(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.axes(a::$ConjArray) = axes_conj(a)
            Base.size(a::$ConjArray) = size_conj(a)
            Base.similar(a::$ConjArray, elt::Type) = similar_conj(a, elt)
            Base.similar(a::$ConjArray, elt::Type, ax) = similar_conj(a, elt, ax)
            Base.similar(a::$ConjArray, elt::Type, ax::Dims) = similar_conj(a, elt, ax)
            Base.copyto!(dest::$AbstractArray, src::$ConjArray) = copyto!_conj(dest, src)
            Base.show(io::IO, a::$ConjArray) = show_conj(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$ConjArray) =
                show_conj(io, mime, a)
        end
    )
end

macro conjarray_broadcast(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$ConjArray) = materialize_conj(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$ConjArray}) =
                BroadcastStyle_conj(arrayt)
        end
    )
end

macro conjarray_stridedviews(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.SV.isstrided(a::$ConjArray) =
                TensorAlgebra.isstrided_conj(a)
            TensorAlgebra.SV.StridedView(a::$ConjArray) =
                TensorAlgebra.StridedView_conj(a)
        end
    )
end

macro conjarray_terminterface(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.iscall(a::$ConjArray) = TensorAlgebra.iscall_conj(a)
            TensorAlgebra.operation(a::$ConjArray) = TensorAlgebra.operation_conj(a)
            TensorAlgebra.arguments(a::$ConjArray) = TensorAlgebra.arguments_conj(a)
        end
    )
end

macro conjarray(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.@conjarray_base $ConjArray $AbstractArray
            TensorAlgebra.@conjarray_broadcast $ConjArray $AbstractArray
            TensorAlgebra.@conjarray_stridedviews $ConjArray $AbstractArray
            TensorAlgebra.@conjarray_terminterface $ConjArray $AbstractArray
        end
    )
end

# Generic constructors, accessors, and properties for AddArrays.
+ₗ(a::AbstractArray, b::AbstractArray) = AddArray(a, b)
addends(a::AbstractArray) = (a,)
addends_type(arrayt::Type{<:AbstractArray}) = Tuple{arrayt}
add_eltype(args::AbstractArray...) = Base.promote_op(+, eltype.(args)...)
add_ndims(args::AbstractArray...) = allequal(ndims, args) ? ndims(first(args)) :
    error("All addends must have the same number of dimensions.")

# Base overloads for AddArrays.
add_axes(args::AbstractArray...) = BC.combine_axes(args...)
axes_add(a::AbstractArray) = add_axes(addends(a)...)
size_add(a::AbstractArray) = length.(axes_add(a))
similar_add(a::AbstractArray) = similar(a, eltype(a))
similar_add(a::AbstractArray, ax::Tuple) = similar(a, eltype(a), ax)
similar_add(a::AbstractArray, elt::Type) = similar(BC.Broadcasted(+, addends(a)), elt)
similar_add(a::AbstractArray, elt::Type, ax) =
    similar(BC.Broadcasted(+, addends(a)), elt, ax)
copyto!_add(dest::AbstractArray, src::AbstractArray) = add!(dest, src, true, false)
show_add(io::IO, a::AbstractArray) = show_lazy(io, a)
show_add(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for AddArrays.
materialize_add(a::AbstractArray) = copy(a)
function BroadcastStyle_add(arrayt::Type{<:AbstractArray})
    args_type = addends_type(arrayt)
    style = Base.promote_op(BC.combine_styles, fieldtypes(args_type)...)()
    return LazyArrayStyle(style)
end

# TensorAlgebra overloads for AddArrays.
function add!_add(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    args = addends(src)
    add!(dest, first(args), α, β)
    for a in Base.tail(args)
        add!(dest, a, α, true)
    end
    return dest
end

# Lazy operations for AddArrays.
added_add(a::AbstractArray, b::AbstractArray) = AddArray((addends(a)..., addends(b)...)...)
mulled_add(α::Number, a::AbstractArray) = +ₗ((α .*ₗ addends(a))...)
## TODO: Define these by expanding all combinations treating both inputs as AddArrays.
## mulled_add(a::AbstractArray, b::AbstractArray) = +ₗ((Ref(a) .*ₗ addends(b))...)
## mulled_add(a::AddArray, b::AbstractArray) = +ₗ((addends(a) .*ₗ Ref(b))...)
## mulled_add(a::AddArray, b::AddArray) = +ₗ((Ref(a) .*ₗ addends(b))...)
conjed_add(a::AbstractArray) = +ₗ(conjed.(addends(a))...)

# TermInterface-like overloads for AddArrays.
iscall_add(::AbstractArray) = true
operation_add(::AbstractArray) = +
arguments_add(a::AbstractArray) = addends(a)

macro addarray_type(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $AddArray{T, N, Args <: Tuple{Vararg{AbstractArray{<:Any, N}}}} <:
                $AbstractArray{T, N}
                args::Args
                function $AddArray(args::AbstractArray...)
                    T = TensorAlgebra.add_eltype(args...)
                    N = TensorAlgebra.add_ndims(args...)
                    return new{T, N, typeof(args)}(args)
                end
            end
            TensorAlgebra.addends(a::$AddArray) = a.args
            TensorAlgebra.addends_type(arrayt::Type{<:$AddArray}) =
                fieldtype(arrayt, :args)
        end
    )
end

macro addarray_base(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.axes(a::$AddArray) = TensorAlgebra.axes_add(a)
            Base.size(a::$AddArray) = TensorAlgebra.size_add(a)
            Base.similar(a::$AddArray) = TensorAlgebra.similar_add(a)
            Base.similar(a::$AddArray, ax::Tuple) = TensorAlgebra.similar_add(a, ax)
            Base.similar(a::$AddArray, elt::Type) = TensorAlgebra.similar_add(a, elt)
            function Base.similar(
                    a::$AddArray, elt::Type,
                    ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
                )
                return TensorAlgebra.similar_add(a, elt, ax)
            end
            Base.similar(a::$AddArray, elt::Type, ax::Dims) =
                TensorAlgebra.similar_add(a, elt, ax)
            Base.similar(a::$AddArray, elt::Type, ax) =
                TensorAlgebra.similar_add(a, elt, ax)
            Base.copyto!(dest::$AbstractArray, src::$AddArray) =
                TensorAlgebra.copyto!_add(dest, src)
            Base.show(io::IO, a::$AddArray) =
                TensorAlgebra.show_add(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$AddArray) =
                TensorAlgebra.show_add(io, mime, a)
        end
    )
end

macro addarray_broadcast(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$AddArray) = TensorAlgebra.materialize_add(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$AddArray}) =
                TensorAlgebra.BroadcastStyle_add(arrayt)
        end
    )
end

macro addarray_lazy(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.:(+ₗ)(a::$AbstractArray, b::$AddArray) =
                TensorAlgebra.added_add(a, b)
            TensorAlgebra.:(+ₗ)(a::$AddArray, b::$AbstractArray) =
                TensorAlgebra.added_add(a, b)
            TensorAlgebra.:(+ₗ)(a::$AddArray, b::$AddArray) =
                TensorAlgebra.added_add(a, b)
            TensorAlgebra.:(*ₗ)(α::Number, a::$AddArray) =
                TensorAlgebra.mulled_add(α, a)
            TensorAlgebra.:(*ₗ)(a::$AbstractArray, b::$AddArray) =
                TensorAlgebra.mulled_add(a, b)
            TensorAlgebra.:(*ₗ)(a::$AddArray, b::$AbstractArray) =
                TensorAlgebra.mulled_add(a, b)
            TensorAlgebra.:(*ₗ)(a::$AddArray, b::$AddArray) =
                TensorAlgebra.mulled_add(a, b)
            TensorAlgebra.conjed(a::$AddArray) =
                TensorAlgebra.conjed_add(a)
        end
    )
end

macro addarray_tensoralgebra(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            function TensorAlgebra.add!(
                    dest::$AbstractArray, src::$AddArray, α::Number, β::Number
                )
                return TensorAlgebra.add!_add(dest, src, α, β)
            end
        end
    )
end

macro addarray_terminterface(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.iscall(a::$AddArray) = TensorAlgebra.iscall_add(a)
            TensorAlgebra.operation(a::$AddArray) = TensorAlgebra.operation_add(a)
            TensorAlgebra.arguments(a::$AddArray) = TensorAlgebra.arguments_add(a)
        end
    )
end

macro addarray(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.@addarray_base $AddArray $AbstractArray
            TensorAlgebra.@addarray_broadcast $AddArray $AbstractArray
            TensorAlgebra.@addarray_lazy $AddArray $AbstractArray
            TensorAlgebra.@addarray_tensoralgebra $AddArray $AbstractArray
            TensorAlgebra.@addarray_terminterface $AddArray $AbstractArray
        end
    )
end

# Generic constructors, accessors, and properties for MulArrays.
*ₗ(a::AbstractArray, b::AbstractArray) = MulArray(a, b)
factors(a::AbstractArray) = (a,)
factor_types(arrayt::Type{<:AbstractArray}) = Base.promote_op(factors, arrayt)
# Same as `LinearAlgebra.matprod`, but duplicated here since it is private.
matprod(x, y) = x * y + x * y
mul_eltype(a::AbstractArray, b::AbstractArray) = Base.promote_op(matprod, eltype(a), eltype(b))
mul_ndims(a::AbstractArray, b::AbstractArray) = ndims(b)
mul_axes(a::AbstractArray, b::AbstractArray) = (axes(a, 1), axes(b, ndims(b)))

# Base overloads for MulArrays.
eltype_mul(a::AbstractArray{T}) where {T} = T
axes_mul(a::AbstractArray) = mul_axes(factors(a)...)
size_mul(a::AbstractArray) = length.(axes_mul(a))
similar_mul(a::AbstractArray) = similar(a, eltype(a))
similar_mul(a::AbstractArray, ax::Tuple) = similar(a, eltype(a), ax)
similar_mul(a::AbstractArray, elt::Type) = similar(a, elt, axes(a))
# TODO: Make use of both arguments to determine the output, maybe
# using `LinearAlgebra.matprod_dest(factors(a)..., elt)`?
similar_mul(a::AbstractArray, elt::Type, ax) = similar(last(factors(a)), elt, ax)
copyto!_mul(dest::AbstractArray, src::AbstractArray) = add!(dest, src, true, false)
show_mul(io::IO, a::AbstractArray) = show_lazy(io, a)
show_mul(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for MulArrays.
materialize_mul(a::AbstractArray) = copy(a)
function BroadcastStyle_mul(arrayt::Type{<:AbstractArray})
    style = Base.promote_op(BC.combine_styles, factor_types(arrayt)...)()
    return LazyArrayStyle(style)
end

# TensorAlgebra overloads for MulArrays.
# We materialize the arguments here to avoid nested lazy evaluation.
# Rewrite rules should make it so that `MulArray` is a "leaf` node of the
# expression tree.
add!_mul(dest::AbstractArray, src::AbstractArray, α::Number, β::Number) =
    LA.mul!(dest, BC.materialize.(factors(src))..., α, β)

# Lazy operations for MulArrays.
conjed_mul(a::AbstractArray) = *ₗ(conjed.(factors(a))...)
# Matmul isn't a broadcasting operation so we materialize (i.e.
# perform the matrix multiplication) when building a broadcast
# expression involving a `MulArray`.
# TODO: Use `Broadcast.broadcastable` interface for this?
to_broadcasted_mul(a::AbstractArray) = *(factors(a)...)

# TermInterface-like overloads for MulArrays.
iscall_mul(::AbstractArray) = true
operation_mul(::AbstractArray) = *
arguments_mul(a::AbstractArray) = factors(a)

macro mularray_type(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $MulArray{T, N, A <: AbstractArray, B <: AbstractArray} <:
                $AbstractArray{T, N}
                a::A
                b::B
                function $MulArray(a::AbstractArray, b::AbstractArray)
                    T = TensorAlgebra.mul_eltype(a, b)
                    N = TensorAlgebra.mul_ndims(a, b)
                    return new{T, N, typeof(a), typeof(b)}(a, b)
                end
            end
            TensorAlgebra.factors(a::$MulArray) = (a.a, a.b)
            TensorAlgebra.factor_types(arrayt::Type{<:$MulArray}) =
                (fieldtype(arrayt, :a), fieldtype(arrayt, :b))
        end
    )
end

macro mularray_base(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.eltype(a::$MulArray) = TensorAlgebra.eltype_mul(a)
            Base.axes(a::$MulArray) = TensorAlgebra.axes_mul(a)
            Base.size(a::$MulArray) = TensorAlgebra.size_mul(a)
            Base.similar(a::$MulArray) = TensorAlgebra.similar_mul(a)
            Base.similar(a::$MulArray, ax::Tuple) = TensorAlgebra.similar_mul(a, ax)
            Base.similar(a::$MulArray, elt::Type) = TensorAlgebra.similar_mul(a, elt)
            Base.similar(
                a::$MulArray, elt::Type,
                ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
            ) = TensorAlgebra.similar_mul(a, elt, ax)
            Base.similar(a::$MulArray, elt::Type, ax) =
                TensorAlgebra.similar_mul(a, elt, ax)
            Base.similar(a::$MulArray, elt::Type, ax::Dims) =
                TensorAlgebra.similar_mul(a, elt, ax)
            Base.copyto!(dest::$AbstractArray, src::$MulArray) =
                TensorAlgebra.copyto!_mul(dest, src)
            Base.show(io::IO, a::$MulArray) = TensorAlgebra.show_mul(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$MulArray) =
                TensorAlgebra.show_mul(io, mime, a)
        end
    )
end

macro mularray_broadcast(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$MulArray) = TensorAlgebra.materialize_mul(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$MulArray}) =
                TensorAlgebra.BroadcastStyle_mul(arrayt)
        end
    )
end

macro mularray_lazy(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.conjed(a::$MulArray) = TensorAlgebra.conjed_mul(a)
            TensorAlgebra.to_broadcasted(a::$MulArray) =
                TensorAlgebra.to_broadcasted_mul(a)
        end
    )
end

macro mularray_tensoralgebra(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            function TensorAlgebra.add!(
                    dest::$AbstractArray, src::$MulArray, α::Number, β::Number
                )
                return TensorAlgebra.add!_mul(dest, src, α, β)
            end
        end
    )
end

macro mularray_terminterface(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.iscall(a::$MulArray) = TensorAlgebra.iscall_mul(a)
            TensorAlgebra.operation(a::$MulArray) = TensorAlgebra.operation_mul(a)
            TensorAlgebra.arguments(a::$MulArray) = TensorAlgebra.arguments_mul(a)
        end
    )
end

macro mularray(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            TensorAlgebra.@mularray_base $MulArray $AbstractArray
            TensorAlgebra.@mularray_broadcast $MulArray $AbstractArray
            TensorAlgebra.@mularray_lazy $MulArray $AbstractArray
            TensorAlgebra.@mularray_tensoralgebra $MulArray $AbstractArray
            TensorAlgebra.@mularray_terminterface $MulArray $AbstractArray
        end
    )
end

# Define types.
@scaledarray_type ScaledArray
@scaledarray ScaledArray
@conjarray_type ConjArray
@conjarray ConjArray
@addarray_type AddArray
@addarray AddArray
@mularray_type MulArray
@mularray MulArray
