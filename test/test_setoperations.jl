using TensorAlgebra: biperms, contract, smallsetdiff, tuple_indexin
using Test: @test, @test_throws, @testset

@testset "smallsetdiff" begin
    # Order-preserving, returning a `Vector`.
    @test smallsetdiff((:i, :j, :k), (:k, :i)) == [:j]
    @test smallsetdiff([:i, :j, :k], [:k, :i]) == [:j]
    # Disjoint and empty cases.
    @test smallsetdiff((:i, :j), ()) == [:i, :j]
    @test smallsetdiff((:i, :j), (:i, :j)) == []
end

@testset "tuple_indexin" begin
    # Position of each element of the first tuple in the second collection.
    @test tuple_indexin((:c, :b), (:a, :b, :c, :d)) == (3, 2)
    @test tuple_indexin((:c, :b), [:a, :b, :c, :d]) == (3, 2)
    @test tuple_indexin((), (:a, :b)) == ()
end

@testset "biperms rejects an inconsistent destination" begin
    # The destination must carry exactly the uncontracted labels.
    @test_throws ArgumentError biperms(contract, (:i, :j, :k), (:i, :j), (:j, :k))
    @test_throws ArgumentError biperms(contract, (:i,), (:i, :j), (:j, :k))
end

@testset "biperms with tuple and vector labels agree" begin
    # A representative multi-index contraction: C_il = A_ijk B_kjl.
    bp = biperms(contract, (:i, :l), (:i, :j, :k), (:k, :j, :l))
    @test biperms(contract, [:i, :l], (:i, :j, :k), (:k, :j, :l)) == bp
    @test biperms(contract, (:i, :l), [:i, :j, :k], [:k, :j, :l]) == bp
end
