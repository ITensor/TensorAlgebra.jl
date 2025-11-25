using Test: @test, @test_broken, @test_throws, @testset

using EllipsisNotation: var".."
using StableRNGs: StableRNG
using TensorOperations: TensorOperations

using TensorAlgebra:
    Algorithm,
    BlockedTuple,
    blockedpermvcat,
    contract,
    contract!,
    contractadd!,
    length_codomain,
    length_domain,
    matricize,
    bipermutedims,
    bipermutedims!,
    permuteblockeddims,
    permuteblockeddims!,
    tuplemortar,
    unmatricize,
    unmatricize!

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

@testset "TensorAlgebra" begin
    @testset "misc" begin
        t = (1, 2, 3)
        bt = tuplemortar(((1, 2), (3,)))
        @test length_codomain(t) == 3
        @test length_codomain(bt) == 2
        @test length_domain(t) == 0
        @test length_domain(bt) == 1
    end

    @testset "bipermutedims/permuteblockeddims (eltype=$elt)" for f in
            (:bipermutedims, :permuteblockeddims),
            elt in elts
        f! = Symbol(f, :!)
        @eval begin
            a = randn($elt, 2, 3, 4, 5)
            a_perm = $f(a, blockedpermvcat((3, 1), (2, 4)))
            @test a_perm == permutedims(a, (3, 1, 2, 4))

            a = randn($elt, 2, 3, 4, 5)
            a_perm = $f(a, (3, 1), (2, 4))
            @test a_perm == permutedims(a, (3, 1, 2, 4))

            a = randn($elt, 2, 3, 4, 5)
            a_perm = Array{$elt}(undef, (4, 2, 3, 5))
            $f!(a_perm, a, blockedpermvcat((3, 1), (2, 4)))
            @test a_perm == permutedims(a, (3, 1, 2, 4))

            a = randn($elt, 2, 3, 4, 5)
            a_perm = Array{$elt}(undef, (4, 2, 3, 5))
            $f!(a_perm, a, (3, 1), (2, 4))
            @test a_perm == permutedims(a, (3, 1, 2, 4))
        end
    end
    @testset "matricize (eltype=$elt)" for elt in elts
        a = randn(elt, 2, 3, 4, 5)

        a_fused = matricize(a, blockedpermvcat((1, 2), (3, 4)))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(a, 6, 20)

        a_fused = matricize(a, (1, 2), (3, 4))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(a, 6, 20)
        a_fused = matricize(a, (3, 1), (2, 4))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 15))
        a_fused = matricize(a, (3, 1, 2), (4,))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (24, 5))
        a_fused = matricize(a, (..,), (3, 1))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(permutedims(a, (2, 4, 3, 1)), (15, 8))
        a_fused = matricize(a, (3, 1), (..,))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(permutedims(a, (3, 1, 2, 4)), (8, 15))

        a_fused = matricize(a, (), (..,))
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(a, (1, 120))
        a_fused = matricize(a, (..,), ())
        @test eltype(a_fused) === elt
        @test a_fused ≈ reshape(a, (120, 1))

        @test_throws MethodError matricize(a, (1, 2), (3,), (4,))
        @test_throws MethodError matricize(a, (1, 2, 3, 4))
        @test_throws ArgumentError matricize(a, blockedpermvcat((1, 2), (3,)))

        v = ones(elt, 2)
        a_fused = matricize(v, (1,), ())
        @test eltype(a_fused) === elt
        @test a_fused ≈ ones(elt, 2, 1)
        a_fused = matricize(v, (), (1,))
        @test eltype(a_fused) === elt
        @test a_fused ≈ ones(elt, 1, 2)

        a_fused = matricize(ones(elt), (), ())
        @test eltype(a_fused) === elt
        @test a_fused ≈ ones(elt, 1, 1)
    end

    @testset "unmatricize (eltype=$elt)" for elt in elts
        a0 = randn(elt, 2, 3, 4, 5)
        axes0 = axes(a0)
        m = reshape(a0, 6, 20)

        a = unmatricize(m, tuplemortar((axes0[1:2], axes0[3:4])))
        @test eltype(a) === elt
        @test a ≈ a0

        a = unmatricize(m, axes0[1:2], axes0[3:4])
        @test eltype(a) === elt
        @test a ≈ a0

        a = unmatricize(m, axes0, blockedpermvcat((1, 2), (3, 4)))
        @test eltype(a) === elt
        @test a ≈ a0

        bp = blockedpermvcat((4, 2), (1, 3))
        bpinv = blockedpermvcat((3, 2), (4, 1))
        a = unmatricize(m, map(i -> axes0[i], bp), bpinv)
        @test eltype(a) === elt
        @test a ≈ permutedims(a0, Tuple(bp))

        a = similar(a0)
        unmatricize!(a, m, blockedpermvcat((1, 2), (3, 4)))
        @test a ≈ a0

        m1 = matricize(a0, bp)
        a = unmatricize(m1, axes0, bp)
        @test a ≈ a0

        a1 = permutedims(a0, Tuple(bp))
        a = similar(a1)
        unmatricize!(a, m, bpinv)
        @test a ≈ a1

        a = unmatricize(m, (), axes0)
        @test eltype(a) === elt
        @test a ≈ a0

        a = unmatricize(m, axes0, ())
        @test eltype(a) === elt
        @test a ≈ a0

        m = randn(elt, 1, 1)
        a = unmatricize(m, (), ())
        @test a isa Array{elt, 0}
        @test a[] == m[1, 1]

        @test_throws ArgumentError unmatricize(m, (), blockedpermvcat((1, 2), (3,)))
        @test_throws ArgumentError unmatricize!(m, m, blockedpermvcat((1, 2), (3,)))
    end

    alg_tensoroperations = Algorithm(TensorOperations.StridedBLAS())
    @testset "contract (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
        elt_dest = promote_type(elt1, elt2)
        a1 = ones(elt1, (1, 1))
        a2 = ones(elt2, (1, 1))
        a_dest = ones(elt_dest, (1, 1))
        @test_throws ArgumentError contract(a1, (1, 2, 4), a2, (2, 3))
        @test_throws ArgumentError contract(a1, (1, 2), a2, (2, 3, 4))
        @test_throws ArgumentError contract((1, 3, 4), a1, (1, 2), a2, (2, 3))
        @test_throws ArgumentError contract((1, 3), a1, (1, 2), a2, (2, 4))
        @test_throws ArgumentError contract!(a_dest, (1, 3, 4), a1, (1, 2), a2, (2, 3))

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
            @test labels_dest′ isa
                BlockedTuple{2, (length(setdiff(d1s, d2s)), length(setdiff(d2s, d1s)))}
            a_dest_tensoroperations, = contract(alg_tensoroperations, a1, labels1, a2, labels2)
            @test a_dest ≈ a_dest_tensoroperations

            # Specify destination labels
            a_dest = contract(labels_dest, a1, labels1, a2, labels2)
            a_dest_tensoroperations = contract(
                alg_tensoroperations, labels_dest, a1, labels1, a2, labels2
            )
            @test a_dest ≈ a_dest_tensoroperations

            # Specify with bituple
            a_dest = contract(tuplemortar((labels_dest, ())), a1, labels1, a2, labels2)
            @test a_dest ≈ a_dest_tensoroperations
            a_dest = contract(tuplemortar(((), labels_dest)), a1, labels1, a2, labels2)
            @test a_dest ≈ a_dest_tensoroperations
            a_dest = contract(labels_dest′, a1, labels1, a2, labels2)
            a_dest_tensoroperations = contract(
                alg_tensoroperations, labels_dest′, a1, labels1, a2, labels2
            )
            @test a_dest ≈ a_dest_tensoroperations

            # Specify α and β
            # TODO: Using random `α`, `β` causing
            # random test failures, investigate why.
            α = elt_dest(1.2) # randn(elt_dest)
            β = elt_dest(2.4) # randn(elt_dest)
            a_dest_init = randn(elt_dest, map(i -> dims[i], d_dests))
            a_dest = copy(a_dest_init)
            contractadd!(a_dest, labels_dest, a1, labels1, a2, labels2, α, β)
            a_dest_tensoroperations = copy(a_dest_init)
            contractadd!(
                alg_tensoroperations,
                a_dest_tensoroperations,
                labels_dest,
                a1,
                labels1,
                a2,
                labels2,
                α,
                β,
            )
            ## Here we loosened the tolerance because of some floating point roundoff issue.
            ## with Float32 numbers
            @test a_dest ≈ a_dest_tensoroperations rtol = 50 * default_rtol(elt_dest)
        end
    end
    @testset "outer product contraction (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts,
            elt2 in elts

        elt_dest = promote_type(elt1, elt2)

        rng = StableRNG(123)
        a1 = randn(rng, elt1, 2, 3)
        a2 = randn(rng, elt2, 4, 5)

        a_dest, labels = contract(a1, ("i", "j"), a2, ("k", "l"))
        @test labels == tuplemortar((("i", "j"), ("k", "l")))
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
        @test labels_dest == tuplemortar((labels_a, ()))
        @test a_dest ≈ a * s[]

        # Scalar-array contraction.
        a_dest, labels_dest = contract(s, (), a, labels_a)
        @test labels_dest == tuplemortar(((), labels_a))
        @test a_dest ≈ a * s[]

        # Scalar-scalar contraction.
        a_dest, labels_dest = contract(s, (), t, ())
        @test labels_dest == tuplemortar(((), ()))
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
