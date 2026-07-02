using Random: Random, AbstractRNG

"""
    similar_map(prototype, [T,] codomain_axes, domain_axes) -> M

Allocate an array shaped as a linear map from `domain_axes` to `codomain_axes`
with element type `T` (defaulting to `eltype(prototype)`), using `prototype` to
determine the array backend. The domain axes are given un-dualized (codomain
facing) and stored dual, so the default is
`similar(prototype, T, (codomain_axes..., conj.(domain_axes)...))`. `conj`
dualizes a graded axis and is a no-op on a dense axis. Backends with map-shaped
storage (e.g. a `TensorMap`) overload this to build the codomain/domain directly.

# Examples

```jldoctest
julia> using TensorAlgebra: similar_map

julia> cod, dom = (Base.OneTo(2), Base.OneTo(3)), (Base.OneTo(4), Base.OneTo(5));

julia> M = similar_map(randn(3), Float32, cod, dom);

julia> eltype(M), size(M)
(Float32, (2, 3, 4, 5))
```
"""
function similar_map(prototype, ::Type{T}, codomain_axes, domain_axes) where {T}
    return similar(prototype, T, (codomain_axes..., conj.(domain_axes)...))
end
function similar_map(prototype, codomain_axes, domain_axes)
    return similar_map(prototype, eltype(prototype), codomain_axes, domain_axes)
end

"""
    zeros([T,] axes) -> A
    randn([rng,] [T,] axes) -> A
    rand([rng,] [T,] axes) -> A

Axis-friendly counterparts of `Base.zeros`/`Base.randn`/`Base.rand`, taking the axes
as a single tuple. `Base.zeros` already accepts axes, but `Base.randn`/`Base.rand`
accept only integer dims, so these fill that gap for dense `Base.OneTo` axes and
otherwise forward to `Base` (so a graded-axis backend that extends `Base.randn`/`rand`
on its axis type is picked up). These are the flat (non-map) companions of
[`zeros_map`](@ref).
"""
function zeros end
function randn end
function rand end
@doc (@doc zeros) randn
@doc (@doc zeros) rand

zeros(::Type{T}, axes::Tuple) where {T} = Base.zeros(T, axes)
for (f, g) in ((:randn, :randn), (:rand, :rand))
    @eval begin
        $f(rng::AbstractRNG, ::Type{T}, axes::Tuple) where {T} = Base.$g(rng, T, axes)
        function $f(
                rng::AbstractRNG, ::Type{T}, axes::Tuple{Base.OneTo, Vararg{Base.OneTo}}
            ) where {T}
            return Base.$g(rng, T, map(length, axes))
        end
    end
end

"""
    zeros_map([T,] codomain_axes, domain_axes) -> M
    randn_map([rng,] [T,] codomain_axes, domain_axes) -> M
    rand_map([rng,] [T,] codomain_axes, domain_axes) -> M

Construct an array shaped as a linear map from `domain_axes` to `codomain_axes`,
filled with zeros (`zeros_map`), normally-distributed values (`randn_map`), or
uniformly-distributed values (`rand_map`), with element type `T` (defaulting to
`Float64`). These are the value-filling companions of [`similar_map`](@ref): the
domain axes are given un-dualized (codomain facing) and stored dual, so the default
flattens to the axis-friendly [`zeros`](@ref)/[`randn`](@ref)/[`rand`](@ref) over
`(codomain_axes..., conj.(domain_axes)...)` (`conj` dualizes a graded axis and is a
no-op on a dense one). Backends with map-shaped storage (e.g. a `TensorMap`) overload
these to build the codomain/domain directly.
"""
function zeros_map end
function randn_map end
function rand_map end
@doc (@doc zeros_map) randn_map
@doc (@doc zeros_map) rand_map

zeros_map(codomain_axes, domain_axes) = zeros_map(Float64, codomain_axes, domain_axes)
function zeros_map(::Type{T}, codomain_axes, domain_axes) where {T}
    return zeros(T, (codomain_axes..., conj.(domain_axes)...))
end

for f in (:randn_map, :rand_map)
    g = Symbol(chopsuffix(String(f), "_map"))
    @eval begin
        $f(codomain_axes, domain_axes) =
            $f(Random.default_rng(), codomain_axes, domain_axes)
        function $f(rng::AbstractRNG, codomain_axes, domain_axes)
            return $f(rng, Float64, codomain_axes, domain_axes)
        end
        function $f(::Type{T}, codomain_axes, domain_axes) where {T}
            return $f(Random.default_rng(), T, codomain_axes, domain_axes)
        end
        function $f(rng::AbstractRNG, ::Type{T}, codomain_axes, domain_axes) where {T}
            return $g(rng, T, (codomain_axes..., conj.(domain_axes)...))
        end
    end
end
