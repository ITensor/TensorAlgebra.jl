# `directsum` — the direct sum along `dims`, parameterized by a `FusionStyle` (the same trait
# `matricize` uses) so the fuse/rotate behavior is selectable. The style for a direct sum over
# several arguments is resolved by folding the per-argument `directsum_style` with the binary
# `directsum_style(s1, s2)` combine, mirroring how `Base.Broadcast` folds `BroadcastStyle`.
#
# `directsum_style` defaults to `ReshapeFusion` for every backend: a straight `cat`
# (block-concatenation), a valid direct sum for a dense or an `AbelianGradedArray` backing
# (concat-order axes, no basis rotation). A backend opts its arrays into a fusing/rotating direct
# sum by overloading `directsum_style(::Type)` (e.g. to `SectorFusion`, which merges and sorts the
# summed sectors onto a single canonical basis). The style can also be passed explicitly to
# override, exactly like `matricize(style, ...)`.
#
# `directsum_style` is deliberately distinct from `FusionStyle(a)`: a graded array reports
# `SectorFusion` there (what `matricize` needs), but its `directsum` default is a plain `cat`. Its
# binary combine is likewise kept independent of `FusionStyle`'s so the two traits can diverge. It
# is also distinct from `cat_style`, which resolves the `Concatenated` broadcast style (which
# `copyto!` placement fires), an orthogonal axis.

# Per-argument direct-sum style.
directsum_style(x) = directsum_style(typeof(x))
directsum_style(::Type{<:AbstractArray}) = ReshapeFusion()

# Binary combine of two styles, associative and commutative like `FusionStyle`'s: two of the same
# style combine to that style, and any mismatch falls back to `ReshapeFusion`.
directsum_style(style1::Style, style2::Style) where {Style <: FusionStyle} = Style()
directsum_style(style1::FusionStyle, style2::FusionStyle) = ReshapeFusion()

# Fold the per-argument styles pairwise with the binary combine.
combine_directsum_style(a) = directsum_style(a)
function combine_directsum_style(a, as...)
    return directsum_style(directsum_style(a), combine_directsum_style(as...))
end

function directsum end
directsum(as...; dims) = directsum(combine_directsum_style(as...), as...; dims)
directsum(::ReshapeFusion, as...; dims) = cat(as...; dims)
