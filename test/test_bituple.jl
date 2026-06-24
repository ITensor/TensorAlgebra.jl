using TensorAlgebra: BiTuple, blockedperm, blockedperm_indexin, blockedpermvcat,
    blocklength, blocklengths, blockpermute, blocks, firstblock, lastblock, permmortar,
    trivialbiperm, tuplemortar
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

@testset "BiTuple (axis bituple)" begin
    bt = @constinferred tuplemortar(((true, 'a'), (2.0,)))
    @test bt isa BiTuple{2, 1}
    @test (@constinferred Tuple(bt)) == (true, 'a', 2.0)
    @test (@constinferred blocks(bt)) == ((true, 'a'), (2.0,))
    @test (@constinferred firstblock(bt)) == (true, 'a')
    @test (@constinferred lastblock(bt)) == (2.0,)
    @test (@constinferred blocklengths(bt)) == (2, 1)
    @test blocklength(bt) == 2
    @test length(bt) == 3
    @test bt[1] == true
    @test bt[3] == 2.0
    @test collect(bt) == [true, 'a', 2.0]

    bt_int = tuplemortar(((1,), (2, 3)))
    @test eltype(bt_int) === Int

    # Empty blocks are allowed.
    bt0 = @constinferred tuplemortar(((1,), ()))
    @test blocks(bt0) == ((1,), ())
    @test blocklengths(bt0) == (1, 0)
    @test Tuple(bt0) == (1,)
end

@testset "BiTuple (biperm)" begin
    p = @constinferred permmortar(((3, 4, 5), (2, 1)))
    @test Tuple(p) === (3, 4, 5, 2, 1)
    @test isperm(Tuple(p))
    @test blocks(p) == ((3, 4, 5), (2, 1))
    @test blocklengths(p) == (3, 2)
    @test p == blockedpermvcat((3, 4, 5), (2, 1))
    @test p == blockedperm((3, 4, 5, 2, 1), (3, 2))
    @test Tuple(@constinferred invperm(p)) == invperm(Tuple(p))

    # The perm builders validate that the flat tuple is a permutation.
    @test_throws AssertionError permmortar(((3, 5), (2, 1)))
    @test_throws AssertionError blockedpermvcat((0, 1), (2, 3))

    # Trivial biperm: identity split into codomain/domain, built type-stably.
    tb = @constinferred trivialbiperm(Val(2), Val(4))
    @test blocks(tb) == ((1, 2), (3, 4))
    @test Tuple(tb) == (1, 2, 3, 4)
    @test blocks(@constinferred trivialbiperm(Val(0), Val(2))) == ((), (1, 2))

    # Locate two label groups within a collection.
    p = blockedperm_indexin(("a", "b", "c", "d"), ("c", "a"), ("b", "d"))
    @test p == blockedpermvcat((3, 1), (2, 4))

    # blockpermute splits a collection according to a biperm.
    bp = blockpermute((10, 20, 30, 40), trivialbiperm(Val(1), Val(4)))
    @test bp isa BiTuple
    @test blocks(bp) == ((10,), (20, 30, 40))
end
