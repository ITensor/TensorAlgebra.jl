using TensorAlgebra: BiTuple, bipartition, biperm, tuple_indexin
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

@testset "BiTuple" begin
    bt = @constinferred BiTuple((true, 'a'), (2.0,))
    @test bt isa BiTuple{2, 1}
    @test bt.t1 == (true, 'a')
    @test bt.t2 == (2.0,)
    @test (@constinferred Tuple(bt)) == (true, 'a', 2.0)
    @test length(bt) == 3
    # Acts like the flat tuple for indexing and iteration.
    @test bt[1] == true
    @test bt[3] == 2.0
    @test collect(bt) == [true, 'a', 2.0]

    @test eltype(BiTuple((1,), (2, 3))) === Int

    # Empty blocks are allowed.
    bt0 = @constinferred BiTuple((1,), ())
    @test bt0.t1 == (1,)
    @test bt0.t2 == ()
    @test Tuple(bt0) == (1,)

    # Split constructor: split a flat tuple at the given codomain length.
    @test (@constinferred BiTuple((3, 4, 5, 2, 1), Val(3))) == BiTuple((3, 4, 5), (2, 1))

    # Equality compares the two blocks.
    @test BiTuple((1, 2), (3,)) == BiTuple((1, 2), (3,))
    @test BiTuple((1, 2), (3,)) != BiTuple((1,), (2, 3))
end

@testset "biperm" begin
    p = BiTuple((3, 4, 5), (2, 1))
    @test Tuple(p) === (3, 4, 5, 2, 1)
    @test isperm(Tuple(p))
    @test (@constinferred invperm(p)) isa BiTuple{3, 2}
    @test Tuple(invperm(p)) == invperm(Tuple(p))

    # `bipartition` splits a flat tuple in place (no permutation).
    @test (@constinferred bipartition((3, 4, 5, 2, 1), Val(3))) == ((3, 4, 5), (2, 1))
    @test bipartition((10, 20), Val(0)) == ((), (10, 20))

    # `tuple_indexin` locates labels within a collection.
    @test tuple_indexin(("c", "a"), ("a", "b", "c", "d")) == (3, 1)

    # `biperm` locates two partitioning groups within a collection.
    @test biperm(("a", "b", "c", "d"), ("c", "b"), ("d", "a")) == ((3, 2), (4, 1))
    # The groups must partition the collection.
    @test_throws ArgumentError biperm(("a", "b", "c", "d"), ("c", "b"), ("a",))

    # `bipartition` splits a collection by a biperm or by two index groups.
    @test bipartition((10, 20, 30, 40), BiTuple((1,), (2, 3, 4))) == ((10,), (20, 30, 40))
    @test bipartition((10, 20, 30, 40), (1,), (2, 3, 4)) == ((10,), (20, 30, 40))
end
