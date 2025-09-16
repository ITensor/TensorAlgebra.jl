using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, biperm
using Test: @test, @testset

@testset "Matrix functions (eltype=$elt)" for elt in (Float32, ComplexF64)
  for f in TensorAlgebra.MATRIX_FUNCTIONS
    f == :cbrt && elt <: Complex && continue
    f == :cbrt && VERSION < v"1.11-" && continue
    @eval begin
      rng = StableRNG(123)
      a = randn(rng, $elt, (2, 2, 2, 2))
      for fa in (
        TensorAlgebra.$f(a, (:a, :b, :c, :d), (:c, :b), (:d, :a)),
        TensorAlgebra.$f(a, biperm((3, 2, 4, 1), Val(2))),
      )
        fa′ = reshape($f(reshape(permutedims(a, (3, 2, 4, 1)), (4, 4))), (2, 2, 2, 2))
        @test fa ≈ fa′
      end
    end
  end
end
