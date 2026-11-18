using Random: Random, AbstractRNG

"""
    similar_map([allocator,] prototype, [T,] axes_codomain, axes_domain) -> M

Allocate an array shaped as a linear map from `axes_domain` to `axes_codomain`
with element type `T` (defaulting to `eltype(prototype)`), using `prototype` to
determine the array backend and `allocator` (defaulting to
[`TensorAlgebra.DefaultAllocator`](@ref)) to provide the memory. The domain axes are given
un-dualized and are stored dualized, so for an `AbstractArray` the result has axes
`(axes_codomain..., conj.(axes_domain)...)`.

Array types overload the form with the allocator, for any allocator, and pass it on to the
allocation of their dense storage. An allocator overloads it for the dense prototypes whose
storage it provides.

# Examples

```jldoctest
julia> using TensorAlgebra: similar_map

julia> cod, dom = (Base.OneTo(2), Base.OneTo(3)), (Base.OneTo(4), Base.OneTo(5));

julia> M = similar_map(randn(3), Float32, cod, dom);

julia> eltype(M), size(M)
(Float32, (2, 3, 4, 5))
```
"""
function similar_map(
        allocator, prototype, ::Type{T}, axes_codomain, axes_domain
    ) where {T}
    return similar(prototype, T, (axes_codomain..., conj.(axes_domain)...))
end
function similar_map(prototype, ::Type{T}, axes_codomain, axes_domain) where {T}
    return similar_map(DefaultAllocator(), prototype, T, axes_codomain, axes_domain)
end
function similar_map(prototype, axes_codomain, axes_domain)
    return similar_map(prototype, eltype(prototype), axes_codomain, axes_domain)
end

"""
    zeros([T,] axes) -> A
    ones([T,] axes) -> A
    randn([rng,] [T,] axes) -> A
    rand([rng,] [T,] axes) -> A
    fill(v, axes) -> A

Construct an array with the given `axes`, a tuple of axes, filled with zeros, ones,
normally-distributed values, uniformly-distributed values, or the value `v`. Unlike
`Base.randn` and `Base.rand`, `randn` and `rand` accept `Base.OneTo` axes as well as integer
dimensions.

See also [`zeros_map`](@ref).
"""
function zeros end
function ones end
function randn end
function rand end
function fill end
@doc (@doc zeros) ones
@doc (@doc zeros) randn
@doc (@doc zeros) rand
@doc (@doc zeros) fill

zeros(::Type{T}, axes::Tuple) where {T} = Base.zeros(T, axes)
ones(::Type{T}, axes::Tuple) where {T} = Base.ones(T, axes)
fill(value, axes::Tuple) = Base.fill(value, axes)
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
    zeros_map([T,] axes_codomain, axes_domain) -> M
    ones_map([T,] axes_codomain, axes_domain) -> M
    randn_map([rng,] [T,] axes_codomain, axes_domain) -> M
    rand_map([rng,] [T,] axes_codomain, axes_domain) -> M
    fill_map(v, axes_codomain, axes_domain) -> M

Construct an array shaped as a linear map from `axes_domain` to `axes_codomain`,
filled with zeros (`zeros_map`), ones (`ones_map`), normally-distributed values
(`randn_map`), uniformly-distributed values (`rand_map`), or the value `v` (`fill_map`),
with element type `T` (defaulting to `Float64`, and taken from `v` for `fill_map`). As with
[`similar_map`](@ref), the domain axes are given un-dualized and are stored dualized.
"""
function zeros_map end
function ones_map end
function randn_map end
function rand_map end
function fill_map end
@doc (@doc zeros_map) ones_map
@doc (@doc zeros_map) randn_map
@doc (@doc zeros_map) rand_map
@doc (@doc zeros_map) fill_map

zeros_map(axes_codomain, axes_domain) = zeros_map(Float64, axes_codomain, axes_domain)
function zeros_map(::Type{T}, axes_codomain, axes_domain) where {T}
    return zeros(T, (axes_codomain..., conj.(axes_domain)...))
end
ones_map(axes_codomain, axes_domain) = ones_map(Float64, axes_codomain, axes_domain)
function ones_map(::Type{T}, axes_codomain, axes_domain) where {T}
    return ones(T, (axes_codomain..., conj.(axes_domain)...))
end
function fill_map(value, axes_codomain, axes_domain)
    return fill(value, (axes_codomain..., conj.(axes_domain)...))
end

for f in (:randn_map, :rand_map)
    g = Symbol(chopsuffix(String(f), "_map"))
    @eval begin
        $f(axes_codomain, axes_domain) =
            $f(Random.default_rng(), axes_codomain, axes_domain)
        function $f(rng::AbstractRNG, axes_codomain, axes_domain)
            return $f(rng, Float64, axes_codomain, axes_domain)
        end
        function $f(::Type{T}, axes_codomain, axes_domain) where {T}
            return $f(Random.default_rng(), T, axes_codomain, axes_domain)
        end
        function $f(rng::AbstractRNG, ::Type{T}, axes_codomain, axes_domain) where {T}
            return $g(rng, T, (axes_codomain..., conj.(axes_domain)...))
        end
    end
end
