# `directsum` — the direct sum along `dims`. For now it is exactly `cat`: block-concatenation with
# concat-order axes and no basis rotation, which is a valid direct sum for dense and
# `AbelianGradedArray` backings alike. A fusing/rotating variant (merging and sorting the summed
# sectors onto one canonical basis) will be added behind a style selector when a backend needs it,
# the same way `matricize` takes a `FusionStyle`.
function directsum end
directsum(as...; dims) = cat(as...; dims)
