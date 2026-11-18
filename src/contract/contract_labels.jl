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

# The labels are replaced by integers before the contraction is derived, so that code compiles
# once for all label types and compares integers. `labels1` becomes `1:n1`, and each label of
# `labels2` reuses its match in `labels1` or else gets `n1 + i2`.
function encode_contraction_labels(labels1, labels2)
    n1 = length(labels1)
    int2 = map(eachindex(labels2)) do i2
        i1 = findfirst(==(labels2[i2]), labels1)
        return isnothing(i1) ? n1 + i2 : i1
    end
    return Base.OneTo(n1), int2
end

# Invert `encode_contraction_labels`.
function decode_contraction_labels(labels_dest, labels1, labels2)
    n1 = length(labels1)
    return map(l -> l <= n1 ? labels1[l] : labels2[l - n1], labels_dest)
end
