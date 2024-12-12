using TensorAlgebra: TensorAlgebra
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  # TODO: fix and re-enable ambiguity checks
  Aqua.test_all(TensorAlgebra; ambiguities=false, piracies=false)
end
