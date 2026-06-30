function contract_labels(a1::AbstractArray, labels1, a2::AbstractArray, labels2)
    return contract_labels(labels1, labels2)
end
function contract_labels(labels1, labels2)
    diff1 = smallsetdiff(labels1, labels2)
    diff2 = smallsetdiff(labels2, labels1)
    return vcat(diff1, diff2)
end

"""
    use_int_labels(labels) -> Bool

Whether to match `labels` to integers before deriving a contraction.

Deriving a contraction makes several passes comparing labels. For label types that are costly to
compare, matching them to integers once and running the rest of the bookkeeping on the integers
is faster. This is `false` by default; a label type with expensive equality opts in by defining

```julia
TensorAlgebra.use_int_labels(::Type{MyLabel}) = true
```
"""
use_int_labels(labels) = use_int_labels(eltype(labels))
use_int_labels(::Type) = false

# Match the labels to integers by equality pattern: `labels1` becomes `1:length(labels1)`, and
# each label of `labels2` reuses operand 1's integer where they match (the contracted labels)
# and otherwise gets a fresh integer `length(labels1) + position`. Encoding the fresh integers
# by position lets `from_int_labels` recover the derived labels by position.
function to_int_labels(labels1, labels2)
    n1 = length(labels1)
    int2 = map(eachindex(labels2)) do i2
        i1 = findfirst(==(labels2[i2]), labels1)
        return isnothing(i1) ? n1 + i2 : i1
    end
    return 1:n1, int2
end

# Invert `to_int_labels`: map integer labels back to the original labels by position.
function from_int_labels(int_labels, labels1, labels2)
    n1 = length(labels1)
    return map(l -> l <= n1 ? labels1[l] : labels2[l - n1], int_labels)
end
