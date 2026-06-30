using LinearAlgebra: Diagonal, diag
using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

@testset "Diagonal TensorAlgebra interface (eltype=$elt)" for elt in (Float64, ComplexF64)
    d = Diagonal(elt[2, 3, 4])

    @testset "bipermutedims preserves Diagonal" begin
        b1 = TensorAlgebra.bipermutedims(d, (1,), (2,))
        @test b1 isa Diagonal
        @test b1 !== d
        @test b1 == d
        # A 2D transpose of a Diagonal is the same Diagonal.
        b2 = TensorAlgebra.bipermutedims(d, (2,), (1,))
        @test b2 isa Diagonal
        @test b2 == d
    end

    @testset "permutedimsop applies the op to the data" begin
        dz = Diagonal(elt <: Complex ? elt[1 + 2im, 3 - im, 2im] : elt[1, 3, 2])
        p = TensorAlgebra.permutedimsop(conj, dz, (1,), (2,))
        @test p isa Diagonal
        @test p == conj(dz)
    end

    @testset "allocate_output returns a Diagonal of the same size" begin
        out = TensorAlgebra.allocate_output(
            TensorAlgebra.permutedimsop, identity, d, (1,), (2,)
        )
        @test out isa Diagonal
        @test size(out) == size(d)
        @test eltype(out) === elt
    end

    @testset "add! accumulates onto a Diagonal" begin
        dest = Diagonal(elt[1, 1, 1])
        # `add!(dest, src, α, β)` computes `α * src + β * dest`.
        TensorAlgebra.add!(dest, d, elt(2), elt(1))
        @test dest isa Diagonal
        @test dest == Diagonal(elt[5, 7, 9])
    end

    @testset "matricize(1, 1) is the identity reshape" begin
        m = TensorAlgebra.matricize(TensorAlgebra.ReshapeFusion(), d, Val(1))
        @test m === d
    end

    @testset "unmatricize round-trips a Diagonal" begin
        ax = axes(d, 1)
        back = TensorAlgebra.unmatricize(TensorAlgebra.ReshapeFusion(), d, (ax,), (ax,))
        @test back === d
    end

    @testset "matrix functions preserve Diagonal" begin
        dp = Diagonal(elt[4, 9, 16])
        s = TensorAlgebra.sqrt(dp, ("i", "j"), ("i",), ("j",))
        @test s isa Diagonal
        @test s ≈ sqrt(dp)
        e = TensorAlgebra.exp(dp, ("i", "j"), ("i",), ("j",))
        @test e isa Diagonal
        @test e ≈ exp(dp)
    end

    @testset "contract densifies (Diagonal is an input structure, not an output one)" begin
        d2 = Diagonal(elt[10, 20, 30])
        # One contracted leg: a matrix product, materialized dense.
        c2, = TensorAlgebra.contract(d, ("i", "k"), d2, ("k", "j"))
        @test !(c2 isa Diagonal)
        @test c2 ≈ d * d2
        # Both legs contracted: a scalar.
        c0, = TensorAlgebra.contract(d, ("i", "j"), d2, ("i", "j"))
        @test ndims(c0) == 0
        @test c0[] ≈ sum(diag(d) .* diag(d2))
        # No contracted legs: a rank-4 outer product.
        c4, = TensorAlgebra.contract(d, ("i", "j"), d2, ("k", "l"))
        @test ndims(c4) == 4
    end
end
