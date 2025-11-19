using BlockArrays: blockfirsts, blocklasts, blocklength, blocklengths, blocks
using EllipsisNotation: var".."
using TensorAlgebra: BlockedPermutation, BlockedTrivialPermutation, BlockedTuple,
    blockedperm, blockedperm_indexin, blockpermute, blockedtrivialperm, blockedpermvcat,
    permmortar, trivialperm, tuplemortar
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred

@testset "BlockedPermutation" begin
    p = @constinferred permmortar(((3, 4, 5), (2, 1)))
    @test Tuple(p) === (3, 4, 5, 2, 1)
    @test isperm(p)
    @test length(p) == 5
    @test blocks(p) == ((3, 4, 5), (2, 1))
    @test blocklength(p) == 2
    @test blocklengths(p) == (3, 2)
    @test blockfirsts(p) == (1, 4)
    @test blocklasts(p) == (3, 5)
    @test p == (@constinferred blockedpermvcat((3, 4, 5), (2, 1)))
    @test p == blockedperm((3, 4, 5, 2, 1), (3, 2))
    @test p == (@constinferred blockedperm((3, 4, 5, 2, 1), Val((3, 2))))
    @test (@constinferred invperm(p)) == blockedpermvcat((5, 4, 1), (2, 3))
    @test p isa BlockedPermutation{2}

    flat = (3, 4, 5, 2, 1)
    @test_throws DimensionMismatch BlockedPermutation{2, (1, 2, 2)}(flat)
    @test_throws DimensionMismatch BlockedPermutation{3, (1, 2, 3)}(flat)
    @test_throws DimensionMismatch BlockedPermutation{3, (-1, 3, 3)}(flat)
    @test_throws AssertionError blockedpermvcat((3, 5), (2, 1))
    @test_throws AssertionError blockedpermvcat((0, 1), (2, 3))
    @test_throws AssertionError blockedpermvcat((0,))
    @test_throws AssertionError blockedpermvcat((2,))

    # Empty block.
    p = @constinferred blockedpermvcat((3, 2), (), (1,))
    @test Tuple(p) === (3, 2, 1)
    @test isperm(p)
    @test length(p) == 3
    @test blocks(p) == ((3, 2), (), (1,))
    @test blocklength(p) == 3
    @test blocklengths(p) == (2, 0, 1)
    @test blockfirsts(p) == (1, 3, 3)
    @test blocklasts(p) == (2, 2, 3)
    @test invperm(p) == blockedpermvcat((3, 2), (), (1,))
    @test p isa BlockedPermutation{3}

    p = @constinferred blockedpermvcat((), ())
    @test Tuple(p) === ()
    @test blocklength(p) == 2
    @test blocklengths(p) == (0, 0)
    @test isperm(p)
    @test length(p) == 0
    @test blocks(p) == ((), ())
    @test p isa BlockedPermutation{2}

    p = @constinferred blockedpermvcat()
    @test Tuple(p) === ()
    @test blocklength(p) == 0
    @test blocklengths(p) == ()
    @test isperm(p)
    @test length(p) == 0
    @test blocks(p) == ()
    @test p isa BlockedPermutation{0}

    p = blockedpermvcat((3, 2), (), (1,))
    bt = tuplemortar(((3, 2), (), (1,)))
    @test (@constinferred BlockedTuple(p)) == bt
    @test (@constinferred map(identity, p)) == bt
    @test (@constinferred p .+ p) == tuplemortar(((6, 4), (), (2,)))
    @test (@constinferred p .+ bt) == tuplemortar(((6, 4), (), (2,)))
    @test (@constinferred bt .+ p) == tuplemortar(((6, 4), (), (2,)))
    @test (@constinferred blockedperm(p)) == p
    @test (@constinferred blockedperm(bt)) == p

    @test_throws ArgumentError blockedpermvcat((1, 3), (2, 4); length = Val(6))

    # Split collection into `BlockedPermutation`.
    p = blockedperm_indexin(("a", "b", "c", "d"), ("c", "a"), ("b", "d"))
    @test p == blockedpermvcat((3, 1), (2, 4))

    # Singleton dimensions.
    p = @constinferred blockedpermvcat((2, 3), 1)
    @test p == blockedpermvcat((2, 3), (1,))

    # First dimensions are unspecified.
    p = blockedpermvcat(.., (4, 3))
    @test p == blockedpermvcat((1,), (2,), (4, 3))
    # Specify length
    p = @constinferred blockedpermvcat(.., (4, 3); length = Val(6))
    @test p == blockedpermvcat((1,), (2,), (5,), (6,), (4, 3))

    # Last dimensions are unspecified.
    p = blockedpermvcat((4, 3), ..)
    @test p == blockedpermvcat((4, 3), (1,), (2,))
    # Specify length
    p = @constinferred blockedpermvcat((4, 3), ..; length = Val(6))
    @test p == blockedpermvcat((4, 3), (1,), (2,), (5,), (6,))

    # Middle dimensions are unspecified.
    p = blockedpermvcat((4, 3), .., 1)
    @test p == blockedpermvcat((4, 3), (2,), (1,))
    # Specify length
    # TODO: Type unstable in Julia 1.12, investigate.
    # @constinferred blockedpermvcat((4, 3), .., 1; length = Val(6))
    p = blockedpermvcat((4, 3), .., 1; length = Val(6))
    @test p == blockedpermvcat((4, 3), (2,), (5,), (6,), (1,))

    # No dimensions are unspecified.
    p = blockedpermvcat((3, 2), .., 1)
    @test p == blockedpermvcat((3, 2), (1,))

    # same with (..,) instead of ..
    p = blockedpermvcat((..,), (4, 3))
    @test p == blockedpermvcat((1, 2), (4, 3))
    p = @constinferred blockedpermvcat((..,), (4, 3); length = Val(6))
    @test p == blockedpermvcat((1, 2, 5, 6), (4, 3))

    p = blockedpermvcat((4, 3), (..,))
    @test p == blockedpermvcat((4, 3), (1, 2))
    p = @constinferred blockedpermvcat((4, 3), (..,); length = Val(6))
    @test p == blockedpermvcat((4, 3), (1, 2, 5, 6))

    p = blockedpermvcat((4, 3), (..,), 1)
    @test p == blockedpermvcat((4, 3), (2,), (1,))
    # TODO: Type unstable in Julia 1.12, investigate.
    # @constinferred blockedpermvcat((4, 3), (..,), 1; length = Val(6))
    p = blockedpermvcat((4, 3), (..,), 1; length = Val(6))
    @test p == blockedpermvcat((4, 3), (2, 5, 6), (1,))

    p = blockedpermvcat((3, 2), (..,), 1)
    @test p == blockedpermvcat((3, 2), (), (1,))

    # blockpermute
    t = (1, 2, 3, 4)
    pblocks = tuplemortar(((4, 3), (), (1, 2)))
    p = blockedperm(pblocks)
    @test (@constinferred blockpermute(t, p)) isa BlockedTuple{3, (2, 0, 2), NTuple{4, Int64}}
    @test blockpermute(t, p) == pblocks
    @test t[p] == pblocks
    @test pblocks[p] == tuplemortar(((2, 1), (), (4, 3)))
    @test p[p] == tuplemortar(((2, 1), (), (4, 3)))
end

@testset "BlockedTrivialPermutation" begin
    tp = blockedtrivialperm((2, 0, 1))

    @test tp isa BlockedTrivialPermutation{3}
    @test Tuple(tp) == (1, 2, 3)
    @test blocklength(tp) == 3
    @test blocklengths(tp) == (2, 0, 1)
    @test trivialperm(blockedpermvcat((3, 2), (), (1,))) == tp

    bt = tuplemortar(((1, 2), (), (3,)))
    @test (@constinferred BlockedTuple(tp)) == bt
    @test (@constinferred blocks(tp)) == blocks(bt)
    @test (@constinferred map(identity, tp)) == bt
    @test (@constinferred tp .+ tp) == tuplemortar(((2, 4), (), (6,)))
    @test (@constinferred tp .+ Tuple(tp)) == tuplemortar(((2, 4), (), (6,)))
    @test (@constinferred tp .+ BlockedTuple(tp)) == tuplemortar(((2, 4), (), (6,)))
    @test (@constinferred blockedperm(tp)) == tp
    @test (@constinferred trivialperm(tp)) == tp
    @test (@constinferred trivialperm(bt)) == tp
end
