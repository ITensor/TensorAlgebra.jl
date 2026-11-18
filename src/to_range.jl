"""
    TensorAlgebra.to_range(space)

Convert a description of a space into a default range type, returning the range unchanged
if it already is one.

  - an `Integer` length -> `Base.OneTo`
  - an existing `AbstractUnitRange` -> itself

Other packages extend this for richer spaces, for example GradedArrays converts a vector of
sector-to-multiplicity pairs into a graded range.
"""
to_range(space::AbstractUnitRange) = space
to_range(space::Integer) = Base.OneTo(space)

"""
    TensorAlgebra.ungrade(r)

Return the plain range underlying the axis `r`, stripping any added structure such as a
grading. A plain `AbstractUnitRange` is returned unchanged. For example, a graded range maps to
the `Base.OneTo` of its total dimension.

# Examples

```jldoctest
julia> import TensorAlgebra

julia> TensorAlgebra.ungrade(2:5)
2:5
```
"""
ungrade(r::AbstractUnitRange) = r
