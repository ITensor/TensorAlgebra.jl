using BlockArrays: Block, blocksize
using BlockSparseArrays: BlockSparseArray
using GradedUnitRanges: dual, gradedrange
using SparseArraysBase: densearray
using SymmetrySectors: U1
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
@testset "`contract` `BlockSparseArray` (eltype=$elt)" for elt in elts
  d = gradedrange([U1(0) => 2, U1(1) => 3])
  a1 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
  a2 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
  a3 = randn_blockdiagonal(elt, (d, dual(d)))
  a1_dense = densearray(a1)
  a2_dense = densearray(a2)
  a3_dense = densearray(a3)

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
  @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
  #=
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
  @test dimnames_dest == dimnames_dest_dense
  @test size(a_dest) == size(a_dest_dense)
  @test a_dest isa BlockSparseArray
  @test a_dest ≈ a_dest_dense
  =#

  # vector vector
  @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
  #=
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
  @test dimnames_dest == dimnames_dest_dense
  @test size(a_dest) == size(a_dest_dense)
  @test a_dest isa BlockSparseArray
  @test a_dest ≈ a_dest_dense
  =#

  # outer product
  @test_broken a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
  #=
  a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
  @test dimnames_dest == dimnames_dest_dense
  @test size(a_dest) == size(a_dest_dense)
  @test a_dest isa BlockSparseArray
  @test a_dest ≈ a_dest_dense
  =#
end
