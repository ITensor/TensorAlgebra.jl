using Test: @test, @test_throws

using BlockArrays: Block, blocklength, blocklengths, blockedrange, blockisequal, blocks

using TensorAlgebra: BlockedTuple

@testset "BlockedTuple" begin
  flat = (1, 'a', 2, 'b', 3)
  divs = (1, 2, 2)

  bt = BlockedTuple{divs}(flat)

  @test Tuple(bt) == flat
  @test bt == BlockedTuple((1,), ('a', 2), ('b', 3))
  @test BlockedTuple(bt) == bt
  @test blocklength(bt) == 3
  @test blocklengths(bt) == (1, 2, 2)
  @test blocks(bt) == ((1,), ('a', 2), ('b', 3))

  @test bt[1] == 1
  @test bt[2] == 'a'
  @test bt[Block(1)] == blocks(bt)[1]
  @test bt[Block(2)] == blocks(bt)[2]
  @test bt[Block(1):Block(2)] == blocks(bt)[1:2]
  @test bt[Block(2)[1:2]] == ('a', 2)
  @test bt[2:4] == ('a', 2, 'b')

  @test firstindex(bt) == 1
  @test lastindex(bt) == 5
  @test length(bt) == 5

  @test iterate(bt) == (1, 2)
  @test iterate(bt, 2) == ('a', 3)
  @test blockisequal(only(axes(bt)), blockedrange([1, 2, 2]))

  @test_throws DimensionMismatch BlockedTuple{(1, 2, 3)}(flat)

  bt = BlockedTuple((1,), (4, 2), (5, 3))
  @test Tuple(bt) == (1, 4, 2, 5, 3)
  @test blocklengths(bt) == (1, 2, 2)
  @test copy(bt) == bt
  @test deepcopy(bt) == bt

  @test map(n -> n + 1, bt) == BlockedTuple{blocklengths(bt)}(Tuple(bt) .+ 1)
  @test bt .+ BlockedTuple((1,), (1, 1), (1, 1)) ==
    BlockedTuple{blocklengths(bt)}(Tuple(bt) .+ 1)
  @test_throws DimensionMismatch bt .+ BlockedTuple((1, 1), (1, 1), (1,))

  bt = BlockedTuple((1:2, 1:2), (1:3,))
  @test length.(bt) == BlockedTuple((2, 2), (3,))
end
