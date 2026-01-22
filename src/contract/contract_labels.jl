function contract_labels(a1::AbstractArray, labels1, a2::AbstractArray, labels2)
    return contract_labels(labels1, labels2)
end
function contract_labels(labels1, labels2)
    diff1 = Tuple(setdiff(labels1, labels2))
    diff2 = Tuple(setdiff(labels2, labels1))
    return tuplemortar((diff1, diff2))
end
