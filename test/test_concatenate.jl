using TensorAlgebra: TensorAlgebra, cat_axes, concatenate, concatenate!, directsum
using Test: @test, @testset

@testset "concatenate" begin
    a = reshape(collect(1.0:6.0), 2, 3)
    b = reshape(collect(7.0:12.0), 2, 3)

    # Single-dim concatenation matches `vcat`/`hcat`.
    @test concatenate(1, a, b) == vcat(a, b)
    @test concatenate(2, a, b) == hcat(a, b)

    # Multi-dim concatenation is the block-diagonal placement (off-diagonal blocks zeroed).
    c = concatenate((1, 2), a, b)
    ref = zeros(4, 6)
    ref[1:2, 1:3] .= a
    ref[3:4, 4:6] .= b
    @test c == ref

    # `concatenate!` writes into a provided destination.
    dest = zeros(4, 6)
    @test concatenate!(dest, (1, 2), a, b) === dest
    @test dest == ref

    # Element type is promoted across all inputs, not taken from the first.
    @test eltype(concatenate(1, a, b .+ 0im)) == ComplexF64

    # `cat_axes` computes the concatenated axes from the arguments.
    @test cat_axes(Val((1, 2)), a, b) == (Base.OneTo(4), Base.OneTo(6))
end

@testset "directsum (forwards to concatenate)" begin
    a = reshape(collect(1.0:6.0), 2, 3)
    b = reshape(collect(7.0:12.0), 2, 3)

    # `directsum` is exactly `concatenate`: block-concatenation, no basis rotation.
    @test directsum(a, b; dims = (1, 2)) == concatenate((1, 2), a, b)
    @test directsum(a, b; dims = 1) == vcat(a, b)

    # N-ary: each summand lands in its own diagonal hyper-block.
    d = reshape(collect(1.0:8.0), 2, 4)
    s = directsum(a, b, d; dims = (1, 2))
    @test size(s) == (6, 10)
    @test s[1:2, 1:3] == a
    @test s[3:4, 4:6] == b
    @test s[5:6, 7:10] == d
end
