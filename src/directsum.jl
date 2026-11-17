# Kept separate from `concatenate` so array types can overload it with a fusing variant.
directsum(dims, as...) = concatenate(dims, as...)
