"""
    similar_map(prototype, [T,] codomain_axes, domain_axes) -> O

Allocate an array shaped as a linear map from `domain_axes` to
`codomain_axes`, with axes `(codomain_axes..., conj.(domain_axes)...)` and
element type `T` (defaulting to `eltype(prototype)`). The domain axes are
conjugated so callers can pass both sides in the same (codomain) direction.
For plain axes `conj` is a no-op. For graded axes it flips the sector arrows.
`prototype` provides the array backend and is not mutated.

# Examples

```jldoctest
julia> using TensorAlgebra: similar_map

julia> O = similar_map(
           randn(3),
           Float32,
           (Base.OneTo(2), Base.OneTo(3)),
           (Base.OneTo(4), Base.OneTo(5))
       );

julia> eltype(O), size(O)
(Float32, (2, 3, 4, 5))
```
"""
function similar_map(prototype, ::Type{T}, codomain_axes, domain_axes) where {T}
    return similar(prototype, T, (codomain_axes..., conj.(domain_axes)...))
end
function similar_map(prototype, codomain_axes, domain_axes)
    return similar_map(prototype, eltype(prototype), codomain_axes, domain_axes)
end
