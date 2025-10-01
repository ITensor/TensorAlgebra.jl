using Test: @testset

using Aqua: Aqua

using TensorAlgebra: TensorAlgebra

@testset "Code quality (Aqua.jl)" begin
    # TODO: fix and re-enable ambiguity checks
    Aqua.test_all(TensorAlgebra; ambiguities = false, piracies = false)
end
