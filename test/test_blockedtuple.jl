using Test: @test, @test_broken, @test_throws, @testset

using BlockArrays:
    Block, BlockVector, blocklength, blocklengths, blockedrange, blockisequal, blocks
using TestExtras: @constinferred

using TensorAlgebra: BlockedTuple, blockeachindex, tuplemortar

@testset "BlockedTuple" begin
    flat = (true, 'a', 2, "b", 3.0)
    divs = (1, 2, 2)

    bt = @constinferred BlockedTuple{3, divs}(flat)
    @test bt isa BlockedTuple{3}
    @test (@constinferred blockeachindex(bt)) == (Block(1), Block(2), Block(3))

    @test (@constinferred Tuple(bt)) == flat
    @test (@constinferred tuplemortar(((true,), ('a', 2), ("b", 3.0)))) == bt
    @test BlockedTuple(flat, divs) == bt
    @test (@constinferred BlockedTuple(bt)) == bt
    @test blocklength(bt) == 3
    @test blocklengths(bt) == (1, 2, 2)
    @test (@constinferred blocks(bt)) == ((true,), ('a', 2), ("b", 3.0))

    @test (@constinferred bt[1]) == true
    @test (@constinferred bt[2]) == 'a'
    @test (@constinferred map(identity, bt)) == bt

    # it is hard to make bt[Block(1)] type stable as compile-time knowledge of 1 is lost in Block
    @test bt[Block(1)] == blocks(bt)[1]
    @test bt[Block(2)] == blocks(bt)[2]
    @test bt[Block(1):Block(2)] == tuplemortar(((true,), ('a', 2)))
    @test bt[Block(2)[1:2]] == ('a', 2)
    @test bt[2:4] == ('a', 2, "b")

    @test firstindex(bt) == 1
    @test lastindex(bt) == 5
    @test length(bt) == 5

    @test iterate(bt) == (1, 2)
    @test iterate(bt, 2) == ('a', 3)
    @test blockisequal(only(axes(bt)), blockedrange([1, 2, 2]))

    @test_throws DimensionMismatch BlockedTuple{2, (1, 2, 2)}(flat)
    @test_throws DimensionMismatch BlockedTuple{3, (1, 2, 3)}(flat)
    @test_throws DimensionMismatch BlockedTuple{3, (-1, 3, 3)}(flat)

    bt = tuplemortar(((1,), (4, 2), (5, 3)))
    @test bt isa BlockedTuple
    @test Tuple(bt) == (1, 4, 2, 5, 3)
    @test blocklengths(bt) == (1, 2, 2)
    @test (@constinferred deepcopy(bt)) == bt

    @test (@constinferred map(n -> n + 1, bt)) ==
        BlockedTuple{3, blocklengths(bt)}(Tuple(bt) .+ 1)
    @test (@constinferred bt .+ tuplemortar(((1,), (1, 1), (1, 1)))) ==
        BlockedTuple{3, blocklengths(bt)}(Tuple(bt) .+ 1)
    @test (@constinferred bt .+ tuplemortar(((1,), (1, 1, 1), (1,)))) isa
        BlockedTuple{4, (1, 2, 1, 1), NTuple{5, Int64}}
    @test bt .+ tuplemortar(((1,), (1, 1, 1), (1,))) ==
        tuplemortar(((2,), (5, 3), (6,), (4,)))

    @test @constinferred(replace(bt, 4 => 40)) ==
        tuplemortar(((1,), (40, 2), (5, 3)))
    @test @constinferred(replace(x -> x == 4 ? 40 : x, bt)) ==
        tuplemortar(((1,), (40, 2), (5, 3)))

    bt = tuplemortar(((1:2, 1:2), (1:3,)))
    @test length.(bt) == tuplemortar(((2, 2), (3,)))
    # TODO: What is this testing? Should we fix it and bring it back?
    @test_broken length.(length.(bt)) == tuplemortar(((1, 1), (1,)))

    bt = tuplemortar(((1,), (2,)))
    @test (@constinferred bt .== bt) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test (bt .== bt) == tuplemortar(((true,), (true,)))
    @test (@constinferred bt .== tuplemortar(((1, 2),))) isa
        BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test (bt .== tuplemortar(((1, 2),))) == tuplemortar(((true,), (true,)))
    @test_throws DimensionMismatch bt .== tuplemortar(((1,), (2,), (3,)))
    @test (@constinferred bt .== (1, 2)) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test (bt .== (1, 2)) == tuplemortar(((true,), (true,)))
    @test_throws DimensionMismatch bt .== (1, 2, 3)
    @test (@constinferred bt .== 1) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test (bt .== 1) == tuplemortar(((true,), (false,)))
    @test (@constinferred bt .== (1,)) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}

    @test (bt .== (1,)) == tuplemortar(((true,), (false,)))
    # BlockedTuple .== AbstractVector is not type stable. Requires fix in BlockArrays
    # TODO: What is this testing? Should we fix it and bring it back?
    @test_broken (bt .== [1, 1]) isa BlockVector{Bool}
    # TODO: What is this testing? Should we fix it and bring it back?
    @test_broken blocks(bt .== [1, 1]) == [[true], [false]]
    @test_throws DimensionMismatch bt .== [1, 2, 3]

    @test (@constinferred (1, 2) .== bt) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test ((1, 2) .== bt) == tuplemortar(((true,), (true,)))
    @test_throws DimensionMismatch (1, 2, 3) .== bt
    @test (@constinferred 1 .== bt) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test (1 .== bt) == tuplemortar(((true,), (false,)))
    @test (@constinferred (1,) .== bt) isa BlockedTuple{2, (1, 1), Tuple{Bool, Bool}}
    @test ((1,) .== bt) == tuplemortar(((true,), (false,)))
    # TODO: What is this testing? Should we fix it and bring it back?
    @test_broken ([1, 1] .== bt) isa BlockVector{Bool}
    # TODO: What is this testing? Should we fix it and bring it back?
    @test_broken blocks([1, 1] .== bt) == [[true], [false]]

    # empty blocks
    bt = tuplemortar(((1,), (), (5, 3)))
    @test bt isa BlockedTuple{3}
    @test Tuple(bt) == (1, 5, 3)
    @test blocklengths(bt) == (1, 0, 2)
    @test (@constinferred blocks(bt)) == ((1,), (), (5, 3))
    @test blockisequal(only(axes(bt)), blockedrange([1, 0, 2]))

    bt = tuplemortar(((), ()))
    @test bt isa BlockedTuple{2}
    @test Tuple(bt) == ()
    @test blocklengths(bt) == (0, 0)
    @test (@constinferred blocks(bt)) == ((), ())
    @test blockisequal(only(axes(bt)), blockedrange([0, 0]))
    @test bt == bt .+ bt

    bt0 = tuplemortar(())
    bt1 = tuplemortar(((),))
    @test bt0 isa BlockedTuple{0}
    @test Tuple(bt0) == ()
    @test blocklengths(bt0) == ()
    @test (@constinferred blocks(bt0)) == ()
    @test blockisequal(only(axes(bt0)), blockedrange(zeros(Int, 0)))
    @test bt0 == bt0
    @test bt != bt1
    @test (@constinferred bt0 .+ bt0) == bt0
    @test (@constinferred bt0 .+ bt1) == bt1
end
