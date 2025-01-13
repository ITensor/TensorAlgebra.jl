using Test: @test, @test_throws

using BlockArrays: Block, blocklength, blocklengths, blockedrange, blockisequal, blocks
using TestExtras: @constinferred

using TensorAlgebra: BlockedTuple, tuplemortar

@testset "BlockedTuple" begin
  flat = (true, 'a', 2, "b", 3.0)
  divs = (1, 2, 2)

  bt = BlockedTuple{divs}(flat)

  @test (@constinferred Tuple(bt)) == flat
  @test bt == tuplemortar(((true,), ('a', 2), ("b", 3.0)))
  @test bt == BlockedTuple(flat, divs)
  @test BlockedTuple(bt) == bt
  @test blocklength(bt) == 3
  @test blocklengths(bt) == (1, 2, 2)
  @test (@constinferred blocks(bt)) == ((true,), ('a', 2), ("b", 3.0))

  @test (@constinferred bt[1]) == true
  @test (@constinferred bt[2]) == 'a'

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

  @test_throws DimensionMismatch BlockedTuple{(1, 2, 3)}(flat)

  bt = tuplemortar(((1,), (4, 2), (5, 3)))
  @test Tuple(bt) == (1, 4, 2, 5, 3)
  @test blocklengths(bt) == (1, 2, 2)
  @test (@constinferred deepcopy(bt)) == bt

  @test (@constinferred map(n -> n + 1, bt)) ==
    BlockedTuple{blocklengths(bt)}(Tuple(bt) .+ 1)
  @test (@constinferred bt .+ tuplemortar(((1,), (1, 1), (1, 1)))) ==
    BlockedTuple{blocklengths(bt)}(Tuple(bt) .+ 1)
  @test_throws DimensionMismatch bt .+ tuplemortar(((1, 1), (1, 1), (1,)))

  bt = tuplemortar(((1:2, 1:2), (1:3,)))
  @test length.(bt) == tuplemortar(((2, 2), (3,)))
end
