# `directsum` is a plain concatenation for now, kept as its own entry point so a fusing/rotating
# variant can later be selected by style, the way `matricize` takes a `MatricizeStyle`.
directsum(dims, as...) = concatenate(dims, as...)
