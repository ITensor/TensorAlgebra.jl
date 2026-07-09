"""
    TensorAlgebra.to_range(space)

Convert a description of a space into a default range type, returning the range unchanged
if it already is one. This lets range and axis constructors accept a space uniformly
instead of each reimplementing the conversion.

  - an `Integer` length -> `Base.OneTo`
  - an existing `AbstractUnitRange` -> itself (idempotent passthrough)

Downstream packages extend this for richer spaces; for example, GradedArrays adds a method
that turns a vector of sector-to-multiplicity pairs into a graded range.
"""
to_range(space::AbstractUnitRange) = space
to_range(space::Integer) = Base.OneTo(space)

"""
    TensorAlgebra.ungrade(r)

Return the ungraded plain range underlying an axis: the range with its block structure,
sectors, charge labels, and arrow/dual direction stripped away, keeping only its extent. On a
plain `AbstractUnitRange` this is the identity (there is nothing graded to strip, and any offset
is preserved). Downstream packages extend it for richer axes: GradedArrays maps a graded range
to the `Base.OneTo` of its total dimension, and a native TensorKit space maps to the `Base.OneTo`
of its dimension.

This is the value that keys equality of named axes, so that a named axis compares equal to its
dual: conjugation flips arrows and charge labels but leaves the ungraded extent unchanged.
"""
ungrade(r::AbstractUnitRange) = r
