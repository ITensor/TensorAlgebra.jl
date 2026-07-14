using TensorAlgebra:
    TensorAlgebra, ReshapeFusion, cat, cat!, cat_axes, concatenate, directsum
using Test: @test, @testset

@testset "cat / concatenate" begin
    a = reshape(collect(1.0:6.0), 2, 3)
    b = reshape(collect(7.0:12.0), 2, 3)

    # Single-dim concatenation matches `Base.cat`.
    @test cat(a, b; dims = 1) == vcat(a, b)
    @test cat(a, b; dims = 2) == hcat(a, b)
    @test concatenate(1, a, b) == vcat(a, b)

    # Multi-dim concatenation is the block-diagonal placement (off-diagonal blocks zeroed).
    c = cat(a, b; dims = (1, 2))
    ref = zeros(4, 6)
    ref[1:2, 1:3] .= a
    ref[3:4, 4:6] .= b
    @test c == ref

    # `cat!` writes into a provided destination.
    dest = zeros(4, 6)
    @test cat!(dest, a, b; dims = (1, 2)) === dest
    @test dest == ref

    # Element type is promoted across all inputs, not taken from the first.
    @test eltype(cat(a, b .+ 0im; dims = 1)) == ComplexF64

    # `cat_axes` computes the concatenated axes from the arguments.
    @test cat_axes(Val((1, 2)), a, b) == (Base.OneTo(4), Base.OneTo(6))
end

@testset "directsum (dense: ReshapeFusion -> cat)" begin
    a = reshape(collect(1.0:6.0), 2, 3)
    b = reshape(collect(7.0:12.0), 2, 3)

    # Dense arrays carry `ReshapeFusion`, so `directsum` is exactly `cat` (no rotation needed).
    @test TensorAlgebra.FusionStyle(a) === ReshapeFusion()
    @test directsum(a, b; dims = (1, 2)) == cat(a, b; dims = (1, 2))
    @test directsum(a, b; dims = 1) == vcat(a, b)

    # N-ary: each summand lands in its own diagonal hyper-block.
    d = reshape(collect(1.0:8.0), 2, 4)
    s = directsum(a, b, d; dims = (1, 2))
    @test size(s) == (6, 10)
    @test s[1:2, 1:3] == a
    @test s[3:4, 4:6] == b
    @test s[5:6, 7:10] == d

    # Passing the style explicitly matches the resolved-from-array path.
    @test directsum(ReshapeFusion(), a, b; dims = (1, 2)) == directsum(a, b; dims = (1, 2))
end
