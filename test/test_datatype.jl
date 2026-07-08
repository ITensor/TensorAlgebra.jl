using LinearAlgebra: Diagonal, transpose
using TensorAlgebra: datatype
using Test: @test, @testset

@testset "datatype" begin
    # A plain array is its own storage type.
    a = randn(2, 3)
    @test datatype(a) === Matrix{Float64}
    @test datatype(typeof(a)) === Matrix{Float64}
    # Instance and type forms agree on the base case.
    @test datatype(a) === datatype(typeof(a))

    # Wrappers recurse through `parent` to the underlying storage, no bespoke method.
    @test datatype(transpose(a)) === Matrix{Float64}
    @test datatype(view(a, 1:2, 1:2)) === Matrix{Float64}
    @test datatype(Diagonal([1.0, 2.0])) === Vector{Float64}

    # A wrapper of a wrapper recurses all the way down; running to completion here is the
    # check that the `parent(x) === x` base case terminates the recursion.
    @test datatype(view(transpose(a), 1:2, 1:2)) === Matrix{Float64}

    # Element type is captured alongside the container.
    @test datatype(randn(ComplexF32, 2)) === Vector{ComplexF32}
end
