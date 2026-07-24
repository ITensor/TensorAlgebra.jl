"""
    isdual(a) -> Bool

Whether the axis `a` is dual, i.e. carries the reversed arrow relative to the
non-dual convention. An ordinary range has no arrow to reverse, so the
`AbstractUnitRange` fallback returns `false`; graded axes and other backends with a
duality concept (e.g. a TensorKit space) override this.

See also [`dual`](@ref).
"""
function isdual end
isdual(::AbstractUnitRange) = false

"""
    dual(a)

The dual of the axis `a`, with its arrow reversed. An ordinary range has no arrow to
reverse, so the `AbstractUnitRange` fallback returns `a` unchanged; graded axes and
other backends with a duality concept (e.g. a TensorKit space) override this.

See also [`isdual`](@ref).
"""
function dual end
dual(a::AbstractUnitRange) = a
