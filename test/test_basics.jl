using EllipsisNotation: var".."
using LinearAlgebra: norm, qr
using StableRNGs: StableRNG
using TensorAlgebra: contract, contract!, fusedims, splitdims
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @testset

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

@testset "TensorAlgebra" begin
  @testset "fusedims (eltype=$elt)" for elt in elts
    a = randn(elt, 2, 3, 4, 5)
    a_fused = fusedims(a, (1, 2), (3, 4))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(a, 6, 20)
    a_fused = fusedims(a, (3, 1), (2, 4))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 15))
    a_fused = fusedims(a, (3, 1, 2), 4)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (24, 5))
    a_fused = fusedims(a, .., (3, 1))
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (2, 4, 3, 1)), (3, 5, 8))
    a_fused = fusedims(a, (3, 1), ..)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 3, 5))
    a_fused = fusedims(a, .., (3, 1), 2)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (4, 3, 1, 2)), (5, 8, 3))
    a_fused = fusedims(a, (3, 1), .., 2)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 4, 2)), (8, 5, 3))
    a_fused = fusedims(a, (3, 1), ..)
    @test eltype(a_fused) === elt
    @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 3, 5))
  end
  @testset "splitdims (eltype=$elt)" for elt in elts
    a = randn(elt, 6, 20)
    a_split = splitdims(a, (2, 3), (5, 4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, (1:2, 1:3), (1:5, 1:4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (5, 4), 1 => (2, 3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (1:5, 1:4), 1 => (1:2, 1:3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 5, 4))
    a_split = splitdims(a, 2 => (5, 4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (6, 5, 4))
    a_split = splitdims(a, 2 => (1:5, 1:4))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (6, 5, 4))
    a_split = splitdims(a, 1 => (2, 3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 20))
    a_split = splitdims(a, 1 => (1:2, 1:3))
    @test eltype(a_split) === elt
    @test a_split ≈ reshape(a, (2, 3, 20))
  end
  using TensorOperations: TensorOperations
  @testset "contract (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
    dims = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    labels = (:a, :b, :c, :d, :e, :f, :g, :h, :i)
    for (d1s, d2s, d_dests) in (
      ((1, 2), (1, 2), ()),
      ((1, 2), (2, 1), ()),
      ((1, 2), (2, 1, 3), (3,)),
      ((1, 2, 3), (2, 1), (3,)),
      ((1, 2), (2, 3), (1, 3)),
      ((1, 2), (2, 3), (3, 1)),
      ((2, 1), (2, 3), (3, 1)),
      ((1, 2, 3), (2, 3, 4), (1, 4)),
      ((1, 2, 3), (2, 3, 4), (4, 1)),
      ((3, 2, 1), (4, 2, 3), (4, 1)),
      ((1, 2, 3), (3, 4), (1, 2, 4)),
      ((1, 2, 3), (3, 4), (4, 1, 2)),
      ((1, 2, 3), (3, 4), (2, 4, 1)),
      ((3, 1, 2), (3, 4), (2, 4, 1)),
      ((3, 2, 1), (4, 3), (2, 4, 1)),
      ((1, 2, 3, 4, 5, 6), (4, 5, 6, 7, 8, 9), (1, 2, 3, 7, 8, 9)),
      ((2, 4, 5, 1, 6, 3), (6, 4, 9, 8, 5, 7), (1, 7, 2, 8, 3, 9)),
    )
      a1 = randn(elt1, map(i -> dims[i], d1s))
      labels1 = map(i -> labels[i], d1s)
      a2 = randn(elt2, map(i -> dims[i], d2s))
      labels2 = map(i -> labels[i], d2s)
      labels_dest = map(i -> labels[i], d_dests)

      # Don't specify destination labels
      a_dest, labels_dest′ = contract(a1, labels1, a2, labels2)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest′, a1, labels1, a2, labels2
      )
      @test a_dest ≈ a_dest_tensoroperations

      # Specify destination labels
      a_dest = contract(labels_dest, a1, labels1, a2, labels2)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest, a1, labels1, a2, labels2
      )
      @test a_dest ≈ a_dest_tensoroperations

      # Specify α and β
      elt_dest = promote_type(elt1, elt2)
      # TODO: Using random `α`, `β` causing
      # random test failures, investigate why.
      α = elt_dest(1.2) # randn(elt_dest)
      β = elt_dest(2.4) # randn(elt_dest)
      a_dest_init = randn(elt_dest, map(i -> dims[i], d_dests))
      a_dest = copy(a_dest_init)
      contract!(a_dest, labels_dest, a1, labels1, a2, labels2, α, β)
      a_dest_tensoroperations = TensorOperations.tensorcontract(
        labels_dest, a1, labels1, a2, labels2
      )
      ## Here we loosened the tolerance because of some floating point roundoff issue.
      ## with Float32 numbers
      @test a_dest ≈ α * a_dest_tensoroperations + β * a_dest_init rtol =
        50 * default_rtol(elt_dest)
    end
  end
  @testset "outer product contraction (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts,
    elt2 in elts

    elt_dest = promote_type(elt1, elt2)

    rng = StableRNG(123)
    a1 = randn(rng, elt1, 2, 3)
    a2 = randn(rng, elt2, 4, 5)

    a_dest, labels = contract(a1, ("i", "j"), a2, ("k", "l"))
    @test labels == ("i", "j", "k", "l")
    @test eltype(a_dest) === elt_dest
    @test a_dest ≈ reshape(vec(a1) * transpose(vec(a2)), (size(a1)..., size(a2)...))

    a_dest = contract(("i", "k", "j", "l"), a1, ("i", "j"), a2, ("k", "l"))
    @test eltype(a_dest) === elt_dest
    @test a_dest ≈ permutedims(
      reshape(vec(a1) * transpose(vec(a2)), (size(a1)..., size(a2)...)), (1, 3, 2, 4)
    )

    a_dest = zeros(elt_dest, 2, 5, 3, 4)
    contract!(a_dest, ("i", "l", "j", "k"), a1, ("i", "j"), a2, ("k", "l"))
    @test a_dest ≈ permutedims(
      reshape(vec(a1) * transpose(vec(a2)), (size(a1)..., size(a2)...)), (1, 4, 2, 3)
    )
  end
  @testset "scalar contraction (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts,
    elt2 in elts

    elt_dest = promote_type(elt1, elt2)

    rng = StableRNG(123)
    a = randn(rng, elt1, (2, 3, 4, 5))
    s = randn(rng, elt2, ())
    t = randn(rng, elt2, ())

    labels_a = ("i", "j", "k", "l")

    # Array-scalar contraction.
    a_dest, labels_dest = contract(a, labels_a, s, ())
    @test labels_dest == labels_a
    @test a_dest ≈ a * s[]

    # Scalar-array contraction.
    a_dest, labels_dest = contract(s, (), a, labels_a)
    @test labels_dest == labels_a
    @test a_dest ≈ a * s[]

    # Scalar-scalar contraction.
    a_dest, labels_dest = contract(s, (), t, ())
    @test labels_dest == ()
    @test a_dest[] ≈ s[] * t[]

    # Specify output labels.
    labels_dest_example = ("j", "l", "i", "k")
    size_dest_example = (3, 5, 2, 4)

    # Array-scalar contraction.
    a_dest = contract(labels_dest_example, a, labels_a, s, ())
    @test size(a_dest) == size_dest_example
    @test a_dest ≈ permutedims(a, (2, 4, 1, 3)) * s[]

    # Scalar-array contraction.
    a_dest = contract(labels_dest_example, s, (), a, labels_a)
    @test size(a_dest) == size_dest_example
    @test a_dest ≈ permutedims(a, (2, 4, 1, 3)) * s[]

    # Scalar-scalar contraction.
    a_dest = contract((), s, (), t, ())
    @test size(a_dest) == ()
    @test a_dest[] ≈ s[] * t[]

    # Array-scalar contraction.
    a_dest = zeros(elt_dest, size_dest_example)
    contract!(a_dest, labels_dest_example, a, labels_a, s, ())
    @test a_dest ≈ permutedims(a, (2, 4, 1, 3)) * s[]

    # Scalar-array contraction.
    a_dest = zeros(elt_dest, size_dest_example)
    contract!(a_dest, labels_dest_example, s, (), a, labels_a)
    @test a_dest ≈ permutedims(a, (2, 4, 1, 3)) * s[]

    # Scalar-scalar contraction.
    a_dest = zeros(elt_dest, ())
    contract!(a_dest, (), s, (), t, ())
    @test a_dest[] ≈ s[] * t[]
  end
end
@testset "qr (eltype=$elt)" for elt in elts
  a = randn(elt, 5, 4, 3, 2)
  labels_a = (:a, :b, :c, :d)
  labels_q = (:b, :a)
  labels_r = (:d, :c)
  q, r = qr(a, labels_a, labels_q, labels_r)
  label_qr = :qr
  a′ = contract(labels_a, q, (labels_q..., label_qr), r, (label_qr, labels_r...))
  @test a ≈ a′
end
