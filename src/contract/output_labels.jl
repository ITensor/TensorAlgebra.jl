function output_labels(
        f::typeof(contract),
        alg::ContractAlgorithm,
        a1::AbstractArray,
        labels1,
        a2::AbstractArray,
        labels2,
    )
    return output_labels(f, alg, labels1, labels2)
end

function output_labels(f::typeof(contract), ::ContractAlgorithm, labels1, labels2)
    return output_labels(f, labels1, labels2)
end

function output_labels(::typeof(contract), labels1, labels2)
    diff1 = Tuple(setdiff(labels1, labels2))
    diff2 = Tuple(setdiff(labels2, labels1))
    return tuplemortar((diff1, diff2))
end
