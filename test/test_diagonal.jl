using LinearAlgebra: Diagonal, diag
using TensorAlgebra: TensorAlgebra
using Test: @test, @test_throws, @testset

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
        m = TensorAlgebra.matricize(TensorAlgebra.ReshapeMatricize(), d, Val(1))
        @test m === d
    end

    @testset "unmatricize round-trips a Diagonal on its own {1,1} axes" begin
        ax = axes(d, 1)
        back = TensorAlgebra.unmatricize(TensorAlgebra.ReshapeMatricize(), d, (ax,), (ax,))
        @test back === d
    end

    @testset "unmatricize densifies a genuine bond-split" begin
        d4 = Diagonal(elt[1, 2, 3, 4])
        codomain_axes = (Base.OneTo(2), Base.OneTo(2))
        domain_axes = (Base.OneTo(4),)
        t = TensorAlgebra.unmatricize(
            TensorAlgebra.ReshapeMatricize(), d4, codomain_axes, domain_axes
        )
        @test !(t isa Diagonal)
        @test t == reshape(Array(d4), 2, 2, 4)
    end

    @testset "unmatricize errors on a mismatched {1,1} split" begin
        wrong = Base.OneTo(length(diag(d)) + 1)
        @test_throws DimensionMismatch TensorAlgebra.unmatricize(
            TensorAlgebra.ReshapeMatricize(), d, (wrong,), (wrong,)
        )
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

    @testset "contract stays Diagonal on the matmul pattern, densifies otherwise" begin
        d2 = Diagonal(elt[10, 20, 30])
        # One contracted leg: the matmul/endomorphism pattern stays Diagonal.
        c2, = TensorAlgebra.contract(d, ("i", "k"), d2, ("k", "j"))
        @test c2 isa Diagonal
        @test c2 ≈ d * d2
        # All transpose variants of the single-contracted-leg pattern stay Diagonal.
        for (l1, l2) in (
                (("i", "k"), ("j", "k")),
                (("k", "i"), ("k", "j")),
                (("k", "i"), ("j", "k")),
            )
            ct, = TensorAlgebra.contract(d, l1, d2, l2)
            @test ct isa Diagonal
            @test ct ≈ d * d2
        end
        # Both legs contracted: a scalar.
        c0, = TensorAlgebra.contract(d, ("i", "j"), d2, ("i", "j"))
        @test ndims(c0) == 0
        @test c0[] ≈ sum(diag(d) .* diag(d2))
        # No contracted legs: a rank-4 outer product, densified.
        c4, = TensorAlgebra.contract(d, ("i", "j"), d2, ("k", "l"))
        @test !(c4 isa Diagonal)
        @test ndims(c4) == 4
        # Diagonal times dense: densifies.
        a = reshape(elt[1:9;], 3, 3)
        cda, = TensorAlgebra.contract(d, ("i", "k"), a, ("k", "j"))
        @test !(cda isa Diagonal)
        @test cda ≈ d * a
    end
end
