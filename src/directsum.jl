# `directsum` is a plain concatenation for now, kept as its own entry point so a fusing/rotating
# variant can later be dispatched on the array type, the way `matricize` is.
directsum(dims, as...) = concatenate(dims, as...)
