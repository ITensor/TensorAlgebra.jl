# `directsum` — the direct sum along `dims`, parameterized by a `FusionStyle` (the same trait
# `matricize` uses) so the fuse/rotate behavior is selectable. The style is resolved by
# `directsum_style`, which defaults to `ReshapeFusion` for every backend: a straight `cat`
# (block-concatenation), a valid direct sum for a dense or an `AbelianGradedArray` backing
# (concat-order axes, no basis rotation). A backend opts its arrays into a fusing/rotating direct
# sum by overloading `directsum_style` (e.g. to `SectorFusion`, which merges+sorts the summed
# sectors onto a single canonical basis) — that path is a deferred opt-in. The style can also be
# passed explicitly to override, exactly like `matricize(style, ...)`.
#
# `directsum_style` is deliberately distinct from `FusionStyle(a)`: a graded array reports
# `SectorFusion` there (what `matricize` needs), but its `directsum` default is a plain `cat`. It is
# also distinct from `cat_style`, which resolves the `Concatenated` *broadcast* style (which
# `copyto!` placement fires), an orthogonal axis.
directsum_style(as...) = ReshapeFusion()

function directsum end
directsum(as...; dims) = directsum(directsum_style(as...), as...; dims)
directsum(::ReshapeFusion, as...; dims) = cat(as...; dims)
