"""
    isdual(a) -> Bool

Returns `true` or `false` depending on if the axis `a` is dual. Falls back to `false`
for `AbstractUnitRange`.
"""
function isdual end
isdual(::AbstractUnitRange) = false

"""
    dual(a)

Returns the dual of the axis `a`. Falls back to returning `a` unchanged for
`AbstractUnitRange`.
"""
function dual end
dual(a::AbstractUnitRange) = a
