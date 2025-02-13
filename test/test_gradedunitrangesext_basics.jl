using BlockArrays: Block
using TensorAlgebra: ⊗
using GradedUnitRanges: GradedUnitRanges, gradedrange, dual, isdual, label
using SymmetrySectors: U1
using Test: @test, @testset

@testset "TensorAlgebraGradedUnitRangesExt" begin
  a1 = gradedrange([U1(0) => 2, U1(1) => 3])
  a2 = gradedrange([U1(2) => 3, U1(3) => 4])
  a = a1 ⊗ a2
  @test label(a[Block(1)]) == U1(2)
  @test label(a[Block(2)]) == U1(3)
  @test label(a[Block(3)]) == U1(3)
  @test label(a[Block(4)]) == U1(4)
  @test a[Block(1)] == 1:6
  @test a[Block(2)] == 7:15
  @test a[Block(3)] == 16:23
  @test a[Block(4)] == 24:35
  @test !isdual(a)

  a = a1 ⊗ dual(a2)
  @test label(a[Block(1)]) == U1(-2)
  @test label(a[Block(2)]) == U1(-1)
  @test label(a[Block(3)]) == U1(-3)
  @test label(a[Block(4)]) == U1(-2)
  @test a[Block(1)] == 1:6
  @test a[Block(2)] == 7:15
  @test a[Block(3)] == 16:23
  @test a[Block(4)] == 24:35
  @test !isdual(a)
end
