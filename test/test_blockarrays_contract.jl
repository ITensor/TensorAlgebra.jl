using BlockArrays: Block, blockedrange, blocksize
using BlockSparseArrays: BlockSparseArray
using SparseArraysBase: densearray
using TensorAlgebra: contract
using TypeParameterAccessors: unspecify_type_parameters
using Random: randn!
using Test: @test, @testset

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
@testset "`contract` `blockedrange` (eltype=$elt)" for elt in elts
  d = blockedrange([2, 3])
  a1_sba = randn_blockdiagonal(elt, (d, d, d, d))
  a2_sba = randn_blockdiagonal(elt, (d, d, d, d))
  a3_sba = randn_blockdiagonal(elt, (d, d))
  a1_dense = densearray(a1_sba)
  a2_dense = densearray(a2_sba)
  a3_dense = densearray(a3_sba)
  a1_block = BlockArray(a1_sba)
  a2_block = BlockArray(a2_sba)
  a3_block = BlockArray(a3_sba)
  a1_blocked = BlockedArray(a1_sba)
  a2_blocked = BlockedArray(a2_sba)
  a3_blocked = BlockedArray(a3_sba)

  # matrix matrix
  a_dest_dense, dimnames_dest_dense = contract(
    a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
  )
  for (a1, a2) in ((a1_block, a2_block), (a1_blocked, a2_blocked), (a1_sba, a2_sba))
    a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa unspecify_type_parameters(typeof(a1))
    @test a_dest ≈ a_dest_dense
  end

  # matrix vector
  a_dest_dense, dimnames_dest_dense = contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
  for (a1, a3) in ((a1_block, a3_block), (a1_blocked, a3_blocked), (a1_sba, a3_sba))
    a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa unspecify_type_parameters(typeof(a1))
    @test a_dest ≈ a_dest_dense
  end

  #  vector matrix
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
  for (a3, a1) in ((a3_block, a1_block), (a3_blocked, a1_blocked), (a3_sba, a1_sba))
    a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa unspecify_type_parameters(typeof(a1))
    @test a_dest ≈ a_dest_dense
  end

  # vector vector
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
  for a3 in (a3_block, a3_blocked, a3_sba)
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa unspecify_type_parameters(typeof(a3))
    @test a_dest ≈ a_dest_dense
  end

  # outer product
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
  for a3 in (a3_block, a3_blocked, a3_sba)
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa unspecify_type_parameters(typeof(a3))
    @test a_dest ≈ a_dest_dense
  end
end
