"""
    similar_map(prototype, [T,] codomain_axes, domain_axes) -> M

Allocate an array shaped as a linear map from `domain_axes` to
`codomain_axes` with element type `T` (defaulting to `eltype(prototype)`),
using `prototype` to determine the array backend. Defaults to
`similar(prototype, T, (codomain_axes..., conj.(domain_axes)...))`.

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
