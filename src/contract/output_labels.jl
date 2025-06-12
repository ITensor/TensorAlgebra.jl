function output_labels(
  f::typeof(contract),
  alg::Algorithm,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
)
  return output_labels(f, alg, labels1, labels2)
end

function output_labels(f::typeof(contract), ::Algorithm, labels1, labels2)
  return output_labels(f, labels1, labels2)
end

function output_labels(::typeof(contract), labels1, labels2)
  diff = symdiff(labels1, labels2)
  return tuplemortar((Tuple(intersect(diff, labels1)), Tuple(intersect(diff, labels2))))
end
