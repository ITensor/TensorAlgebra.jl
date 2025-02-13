using Test: @test, @test_broken, @testset

using BlockArrays: blockfirsts, blocklasts, blocklength, blocklengths, blocks
using EllipsisNotation: var".."
using TestExtras: @constinferred

using TensorAlgebra:
  BlockedPermutation,
  BlockedTrivialPermutation,
  BlockedTuple,
  blockedperm,
  blockedperm_indexin,
  blockedtrivialperm,
  trivialperm

@testset "BlockedPermutation" begin
  p = @constinferred blockedperm((3, 4, 5), (2, 1))
  @test Tuple(p) === (3, 4, 5, 2, 1)
  @test isperm(p)
  @test length(p) == 5
  @test blocks(p) == ((3, 4, 5), (2, 1))
  @test blocklength(p) == 2
  @test blocklengths(p) == (3, 2)
  @test blockfirsts(p) == (1, 4)
  @test blocklasts(p) == (3, 5)
  @test (@constinferred invperm(p)) == blockedperm((5, 4, 1), (2, 3))
  @test p isa BlockedPermutation{2}

  flat = (3, 4, 5, 2, 1)
  @test_throws DimensionMismatch BlockedPermutation{2,(1, 2, 2)}(flat)
  @test_throws DimensionMismatch BlockedPermutation{3,(1, 2, 3)}(flat)
  @test_throws DimensionMismatch BlockedPermutation{3,(-1, 3, 3)}(flat)

  # Empty block.
  p = @constinferred blockedperm((3, 2), (), (1,))
  @test Tuple(p) === (3, 2, 1)
  @test isperm(p)
  @test length(p) == 3
  @test blocks(p) == ((3, 2), (), (1,))
  @test blocklength(p) == 3
  @test blocklengths(p) == (2, 0, 1)
  @test blockfirsts(p) == (1, 3, 3)
  @test blocklasts(p) == (2, 2, 3)
  @test invperm(p) == blockedperm((3, 2), (), (1,))
  @test p isa BlockedPermutation{3}

  p = @constinferred blockedperm((), ())
  @test Tuple(p) === ()
  @test blocklength(p) == 2
  @test blocklengths(p) == (0, 0)
  @test isperm(p)
  @test length(p) == 0
  @test blocks(p) == ((), ())
  @test p isa BlockedPermutation{2}

  p = @constinferred blockedperm()
  @test Tuple(p) === ()
  @test blocklength(p) == 0
  @test blocklengths(p) == ()
  @test isperm(p)
  @test length(p) == 0
  @test blocks(p) == ()
  @test p isa BlockedPermutation{0}

  p = blockedperm((3, 2), (), (1,))
  bt = BlockedTuple{3,(2, 0, 1)}((3, 2, 1))
  @test (@constinferred BlockedTuple(p)) == bt
  @test (@constinferred map(identity, p)) == bt
  @test (@constinferred p .+ p) == BlockedTuple{3,(2, 0, 1)}((6, 4, 2))
  @test (@constinferred blockedperm(p)) == p
  @test (@constinferred blockedperm(bt)) == p

  # Split collection into `BlockedPermutation`.
  p = blockedperm_indexin(("a", "b", "c", "d"), ("c", "a"), ("b", "d"))
  @test p == blockedperm((3, 1), (2, 4))

  # Singleton dimensions.
  p = @constinferred blockedperm((2, 3), 1)
  @test p == blockedperm((2, 3), (1,))

  # First dimensions are unspecified.
  p = blockedperm(.., (4, 3))
  @test p == blockedperm((1, 2), (4, 3))
  # Specify length
  p = @constinferred blockedperm(.., (4, 3); length=Val(6))
  @test p == blockedperm((1, 2, 5, 6), (4, 3))

  # Last dimensions are unspecified.
  p = blockedperm((4, 3), ..)
  @test p == blockedperm((4, 3), (1, 2))
  # Specify length
  p = @constinferred blockedperm((4, 3), ..; length=Val(6))
  @test p == blockedperm((4, 3), (1, 2, 5, 6))

  # Middle dimensions are unspecified.
  p = blockedperm((4, 3), .., 1)
  @test p == blockedperm((4, 3), (2,), (1,))
  # Specify length
  p = @constinferred blockedperm((4, 3), .., 1; length=Val(6))
  @test p == blockedperm((4, 3), (2, 5, 6), (1,))

  # No dimensions are unspecified.
  p = blockedperm((3, 2), .., 1)
  @test p == blockedperm((3, 2), (), (1,))
end

@testset "BlockedTrivialPermutation" begin
  tp = blockedtrivialperm((2, 0, 1))

  @test tp isa BlockedTrivialPermutation{3}
  @test Tuple(tp) == (1, 2, 3)
  @test blocklength(tp) == 3
  @test blocklengths(tp) == (2, 0, 1)
  @test trivialperm(blockedperm((3, 2), (), (1,))) == tp

  bt = BlockedTuple{3,(2, 0, 1)}((1, 2, 3))
  @test (@constinferred BlockedTuple(tp)) == bt
  @test (@constinferred blocks(tp)) == blocks(bt)
  @test (@constinferred map(identity, tp)) == bt
  @test (@constinferred tp .+ tp) == BlockedTuple{3,(2, 0, 1)}((2, 4, 6))
  @test (@constinferred blockedperm(tp)) == tp
  @test (@constinferred trivialperm(tp)) == tp
  @test (@constinferred trivialperm(bt)) == tp
end
