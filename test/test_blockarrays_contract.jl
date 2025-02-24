using BlockArrays: Block, BlockArray, BlockedArray, blockedrange, blocksize
using BlockSparseArrays: BlockSparseArray
using SparseArraysBase: densearray
using TensorAlgebra: contract
using Random: randn!
using Test: @test, @test_broken, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
  a = BlockSparseArray{elt}(axes)
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
  a1_sba = randn_blockdiagonal(elt, (d, d, d, d))
  a2_sba = randn_blockdiagonal(elt, (d, d, d, d))
  a3_sba = randn_blockdiagonal(elt, (d, d))
  a1_dense = densearray(a1_sba)
  a2_dense = densearray(a2_sba)
  a3_dense = densearray(a3_sba)

  @testset "BlockArray" begin
    a1 = BlockArray(a1_sba)
    a2 = BlockArray(a2_sba)
    a3 = BlockArray(a3_sba)

    # matrix matrix
    @test_broken a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
    #=
    a_dest_dense, dimnames_dest_dense = contract(
      a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
    )
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockArray
    @test a_dest ≈ a_dest_dense
    =#

    # matrix vector
    @test_broken a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockArray
    @test a_dest ≈ a_dest_dense
    =#

    #  vector matrix
    @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockArray
    @test a_dest ≈ a_dest_dense
    =#

    # vector vector
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test_broken a_dest isa BlockArray   # TBD relax to AbstractArray{elt,0}?
    @test a_dest ≈ a_dest_dense

    # outer product
    @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockArray
    @test a_dest ≈ a_dest_dense
    =#
  end

  @testset "BlockedArray" begin
    a1 = BlockedArray(a1_sba)
    a2 = BlockedArray(a2_sba)
    a3 = BlockedArray(a3_sba)

    # matrix matrix
    a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
    a_dest_dense, dimnames_dest_dense = contract(
      a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
    )
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockedArray
    @test a_dest ≈ a_dest_dense

    # matrix vector
    @test_broken a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockedArray
    @test a_dest ≈ a_dest_dense
    =#

    #  vector matrix
    @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockedArray
    @test a_dest ≈ a_dest_dense
    =#

    # vector vector
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test_broken a_dest isa BlockedArray   # TBD relax to AbstractArray{elt,0}?
    @test a_dest ≈ a_dest_dense

    # outer product
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockedArray
    @test a_dest ≈ a_dest_dense
  end

  @testset "BlockSparseArray" begin
    a1, a2, a3 = a1_sba, a2_sba, a3_sba

    # matrix matrix
    a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
    a_dest_dense, dimnames_dest_dense = contract(
      a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
    )
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense

    # matrix vector
    @test_broken a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense
    =#

    #  vector matrix
    a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense

    # vector vector
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense

    # outer product
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense
  end
end
