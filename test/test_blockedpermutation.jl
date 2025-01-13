using Test: @test, @test_broken, @testset

using BlockArrays: blockfirsts, blocklasts, blocklength, blocklengths, blocks
using Combinatorics: permutations

using TensorAlgebra: BlockedTuple, blockedperm, blockedperm_indexin

@testset "BlockedPermutation" begin
  p = blockedperm((3, 4, 5), (2, 1))
  @test Tuple(p) === (3, 4, 5, 2, 1)
  @test isperm(p)
  @test length(p) == 5
  @test blocks(p) == ((3, 4, 5), (2, 1))
  @test blocklength(p) == 2
  @test blocklengths(p) == (3, 2)
  @test blockfirsts(p) == (1, 4)
  @test blocklasts(p) == (3, 5)
  @test invperm(p) == blockedperm((5, 4, 1), (2, 3))

  # Empty block.
  p = blockedperm((3, 2), (), (1,))
  @test Tuple(p) === (3, 2, 1)
  @test isperm(p)
  @test length(p) == 3
  @test blocks(p) == ((3, 2), (), (1,))
  @test blocklength(p) == 3
  @test blocklengths(p) == (2, 0, 1)
  @test blockfirsts(p) == (1, 3, 3)
  @test blocklasts(p) == (2, 2, 3)
  @test invperm(p) == blockedperm((3, 2), (), (1,))
  @test BlockedTuple(p) == BlockedTuple{(2, 0, 1)}((3, 2, 1))

  # Split collection into `BlockedPermutation`.
  p = blockedperm_indexin(("a", "b", "c", "d"), ("c", "a"), ("b", "d"))
  @test p == blockedperm((3, 1), (2, 4))

  # Singleton dimensions.
  p = blockedperm((2, 3), 1)
  @test p == blockedperm((2, 3), (1,))

  # First dimensions are unspecified.
  p = blockedperm(.., (4, 3))
  @test p == blockedperm(1, 2, (4, 3))
  # Specify length
  p = blockedperm(.., (4, 3); length=Val(6))
  @test p == blockedperm(1, 2, 5, 6, (4, 3))

  # Last dimensions are unspecified.
  p = blockedperm((4, 3), ..)
  @test p == blockedperm((4, 3), 1, 2)
  # Specify length
  p = blockedperm((4, 3), ..; length=Val(6))
  @test p == blockedperm((4, 3), 1, 2, 5, 6)

  # Middle dimensions are unspecified.
  p = blockedperm((4, 3), .., 1)
  @test p == blockedperm((4, 3), 2, 1)
  # Specify length
  p = blockedperm((4, 3), .., 1; length=Val(6))
  @test p == blockedperm((4, 3), 2, 5, 6, 1)

  # No dimensions are unspecified.
  p = blockedperm((3, 2), .., 1)
  @test p == blockedperm((3, 2), 1)
end
