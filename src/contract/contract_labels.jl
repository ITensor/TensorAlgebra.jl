function contract_labels(a1::AbstractArray, labels1, a2::AbstractArray, labels2)
    return contract_labels(labels1, labels2)
end
function contract_labels(labels1, labels2)
    diff1 = smallsetdiff(labels1, labels2)
    diff2 = smallsetdiff(labels2, labels1)
    return vcat(diff1, diff2)
end

"""
    label_type(labels) -> Type
    label_type(::Type{L}) -> Type

The label type to use when deriving a contraction.

Deriving a contraction makes several passes comparing labels, so a label type that is costly to
compare can map here to a cheaper integer type: the labels are matched to integers by equality
pattern, the bookkeeping runs on the integers, and the derived labels are mapped back. The default
is the identity `label_type(::Type{L}) = L`, so the bookkeeping runs on the labels as-is. A label
type with expensive equality opts in by defining e.g.

```julia
TensorAlgebra.label_type(::Type{MyLabel}) = Int
```
"""
label_type(labels) = label_type(eltype(labels))
label_type(::Type{T}) where {T} = T

# Convert the operands' labels to `label_type`, a no-op when it already matches the labels. When
# it differs (an integer type) the labels are matched to integers by equality pattern: `labels1`
# becomes `1:length(labels1)`, and each of `labels2`'s labels reuses operand 1's integer where
# they match (the contracted labels) and otherwise gets a fresh integer `length(labels1) +
# position`. Encoding the fresh integers by position lets `decode_contraction_labels` map the
# derived labels back.
function encode_contraction_labels(labels1, labels2)
    I = label_type(eltype(labels1))
    I === eltype(labels1) && return labels1, labels2
    n1 = length(labels1)
    int2 = map(eachindex(labels2)) do i2
        i1 = findfirst(==(labels2[i2]), labels1)
        return isnothing(i1) ? I(n1 + i2) : I(i1)
    end
    return Base.OneTo(I(n1)), int2
end

# Invert `encode_contraction_labels`: a no-op when the labels were not converted, otherwise map
# the integer labels back to the original labels by position.
function decode_contraction_labels(labels_dest, labels1, labels2)
    label_type(eltype(labels1)) === eltype(labels1) && return labels_dest
    n1 = length(labels1)
    return map(l -> l <= n1 ? labels1[l] : labels2[l - n1], labels_dest)
end
