using BlockArrays: Block, BlockArray, BlockedArray, blockedrange, blocksize
using Random: randn!
using TensorAlgebra: contract, matricize, unmatricize
using Test: @test, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
    a = zeros(elt, axes)
    blockdiaglength = minimum(blocksize(a))
    for i in 1:blockdiaglength
        b = Block(ntuple(Returns(i), ndims(a)))
        a[b] = randn!(a[b])
    end
    return a
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "`contract` blocked arrays (eltype=$elt)" for elt in elts
    d = blockedrange([2, 3])
    a1 = randn_blockdiagonal(elt, (d, d, d, d))
    a2 = randn_blockdiagonal(elt, (d, d, d, d))
    a3 = randn_blockdiagonal(elt, (d, d))
    a1_dense = convert(Array, a1)
    a2_dense = convert(Array, a2)
    a3_dense = convert(Array, a3)

    @testset "BlockedArray" begin
        # matrix matrix
        a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
        a_dest_dense, dimnames_dest_dense = contract(
            a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
        )
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockedArray{elt}
        @test a_dest ≈ a_dest_dense

        # matrix vector
        a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
        a_dest_dense, dimnames_dest_dense =
            contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockedArray{elt}
        @test a_dest ≈ a_dest_dense

        # vector matrix
        a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
        a_dest_dense, dimnames_dest_dense =
            contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockedArray{elt}
        @test a_dest ≈ a_dest_dense

        # vector vector
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
        a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockedArray{elt, 0}
        @test a_dest ≈ a_dest_dense

        # outer product
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
        a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockedArray{elt}
        @test a_dest ≈ a_dest_dense
    end

    @testset "BlockArray" begin
        a1, a2, a3 = BlockArray.((a1, a2, a3))

        # matrix matrix
        a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
        m1 = matricize(a1, (2, 4), (1, 3))
        m2 = matricize(a2, (3, 1), (2, 4))
        m_dest = matricize(a_dest, Val(2))
        @test m_dest ≈ m1 * m2

        # matrix vector
        a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
        m1 = matricize(a1, (2, 3), (1, 4))
        m2 = matricize(a3, (2, 1), ())
        m_dest = matricize(a_dest, Val(2))
        @test m_dest ≈ m1 * m2

        # vector matrix
        a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
        m1 = matricize(a3, (), (1, 2))
        m2 = matricize(a1, (4, 1), (2, 3))
        m_dest = matricize(a_dest, Val(0))
        @test m_dest ≈ m1 * m2

        # vector vector
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
        m1 = matricize(a3, (), (1, 2))
        m2 = matricize(a3, (2, 1), ())
        m_dest = matricize(a_dest, Val(0))
        @test m_dest ≈ m1 * m2

        # outer product
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
        m1 = matricize(a3, (1, 2), ())
        m2 = matricize(a3, (), (1, 2))
        m_dest = matricize(a_dest, Val(2))
        @test m_dest ≈ m1 * m2
    end
end
