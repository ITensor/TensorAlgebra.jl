function contract_labels(a1::AbstractArray, labels1, a2::AbstractArray, labels2)
    return contract_labels(labels1, labels2)
end
function contract_labels(labels1, labels2)
    r = Vector{promote_type(eltype(labels1), eltype(labels2))}()
    for x in labels1
        x ∉ labels2 && push!(r, x)
    end
    for x in labels2
        x ∉ labels1 && push!(r, x)
    end
    return r
end

"""
    label_type(::Type{L}) -> Type

The label type used internally when deriving a contraction from labels of type `L`. Defaults
to `L`. A label type that is expensive to compare can map to an integer type, which makes the
contraction compare integers instead, for example

```julia
TensorAlgebra.label_type(::Type{MyLabel}) = Int
```
"""
label_type(::Type{T}) where {T} = T

# `labels1` becomes `1:n1`, and each label of `labels2` reuses its match in `labels1` or else gets
# `n1 + i2`. `decode_contraction_labels` relies on this positional encoding.
function encode_contraction_labels(labels1, labels2)
    L = label_type(eltype(labels1))
    L === eltype(labels1) && return labels1, labels2
    n1 = length(labels1)
    int2 = map(eachindex(labels2)) do i2
        i1 = findfirst(==(labels2[i2]), labels1)
        return isnothing(i1) ? L(n1 + i2) : L(i1)
    end
    return Base.OneTo(L(n1)), int2
end

# Invert `encode_contraction_labels`.
function decode_contraction_labels(labels_dest, labels1, labels2)
    label_type(eltype(labels1)) === eltype(labels1) && return labels_dest
    n1 = length(labels1)
    return map(l -> l <= n1 ? labels1[l] : labels2[l - n1], labels_dest)
end
