@eval module $(gensym())
using Test: @testset

@testset "TensorAlgebra.jl" begin
  filenames = filter(readdir(@__DIR__)) do f
    startswith("test_")(f) && endswith(".jl")(f)
  end
  @testset "Test $filename" for filename in filenames
    include(filename)
  end
end
end
