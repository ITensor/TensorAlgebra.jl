function extract_contract_labels(contraction::AbstractString)
  symbolsC = match(r"C\[([^\]]*)\]", contraction)
  labelsC = split(symbolsC.captures[1], ","; keepempty=false)
  symbolsA = match(r"A\[([^\]]*)\]", contraction)
  labelsA = split(symbolsA.captures[1], ","; keepempty=false)
  symbolsB = match(r"B\[([^\]]*)\]", contraction)
  labelsB = split(symbolsB.captures[1], ","; keepempty=false)
  return labelsC, labelsA, labelsB
end

function generate_contract_benchmark(
  line::AbstractString; T=Float64, alg=default_contract_alg(), do_alpha=true, do_beta=true
)
  line_split = split(line, " & ")
  @assert length(line_split) == 2 "Invalid line format:\n$line"
  contraction, sizes = line_split

  # extract labels
  labelsC, labelsA, labelsB = map(Tuple, extract_contract_labels(contraction))
  #   pA, pB, pC = TensorOperations.contract_indices(
  #     tuple(labelsA...), tuple(labelsB...), tuple(labelsC...)
  #   )

  # extract sizes
  subsizes = Dict{String,Int}()
  for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref(":"))
    subsizes[label] = parse(Int, sz)
  end
  szA = getindex.(Ref(subsizes), labelsA)
  szB = getindex.(Ref(subsizes), labelsB)
  szC = getindex.(Ref(subsizes), labelsC)
  setup_tensors() = (rand(T, szA...), rand(T, szB...), rand(T, szC...))

  if do_alpha && do_beta
    α, β = rand(T, 2)
    return @benchmarkable(
      contract!($alg, C, $labelsC, A, $labelsA, B, $labelsB, $α, $β),
      setup = ((A, B, C) = $setup_tensors()),
      evals = 1
    )
  elseif do_alpha
    α = rand(T)
    return @benchmarkable(
      contract!($alg, C, $labelsC, A, $labelsA, B, $labelsB, $α),
      setup = ((A, B, C) = $setup_tensors()),
      evals = 1
    )
  elseif do_beta
    β = rand(T)
    return @benchmarkable(
      contract!($alg, C, $labelsC, A, $labelsA, B, $labelsB, true, $β),
      setup = ((A, B, C) = $setup_tensors()),
      evals = 1
    )
  else
    return @benchmarkable(
      contract!($alg, C, $labelsC, A, $labelsA, B, $labelsB),
      setup = ((A, B, C) = $setup_tensors()),
      evals = 1
    )
  end
end

function compute_contract_ops(line::AbstractString)
  line_split = split(line, " & ")
  @assert length(line_split) == 2 "Invalid line format:\n$line"
  _, sizes = line_split

  # extract sizes
  subsizes = Dict{String,Int}()
  for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
    subsizes[label] = parse(Int, sz)
  end
  return prod(collect(values(subsizes)))
end
