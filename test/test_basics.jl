import TensorAlgebra
using EllipsisNotation: var".."
using StableRNGs: StableRNG
using TensorAlgebra: BiTuple, ContractAlgorithm, bipermutedims, bipermutedims!, contract,
    contract!, contractadd!, length_codomain, length_domain, matricize, unmatricize,
    unmatricize!
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @test_throws, @testset

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

# A label type that opts into integer relabeling, to exercise the contraction path that matches
# labels to integers before deriving the contraction.
struct OptInLabel
    id::Int
end
TensorAlgebra.label_type(::Type{OptInLabel}) = Int

@testset "TensorAlgebra" begin
    @testset "misc" begin
        t = (1, 2, 3)
        bt = BiTuple((1, 2), (3,))
        @test length_codomain(t) == 3
        @test length_codomain(bt) == 2
        @test length_domain(t) == 0
        @test length_domain(bt) == 1
    end

    @testset "trivialrange" begin
        @test TensorAlgebra.trivialrange(Base.OneTo{Int}) === Base.OneTo(1)
        @test TensorAlgebra.trivialrange(Base.OneTo(5)) === Base.OneTo(1)
    end

    @testset "bipermutedims (eltype=$elt)" for elt in elts
        a = randn(elt, 2, 3, 4, 5)
        a_perm = bipermutedims(a, BiTuple((3, 1), (2, 4)))
        @test a_perm == permutedims(a, (3, 1, 2, 4))

        a = randn(elt, 2, 3, 4, 5)
        a_perm = bipermutedims(a, (3, 1), (2, 4))
        @test a_perm == permutedims(a, (3, 1, 2, 4))

        a = randn(elt, 2, 3, 4, 5)
        a_perm = Array{elt}(undef, (4, 2, 3, 5))
        bipermutedims!(a_perm, a, BiTuple((3, 1), (2, 4)))
        @test a_perm == permutedims(a, (3, 1, 2, 4))

        a = randn(elt, 2, 3, 4, 5)
        a_perm = Array{elt}(undef, (4, 2, 3, 5))
        bipermutedims!(a_perm, a, (3, 1), (2, 4))
        @test a_perm == permutedims(a, (3, 1, 2, 4))
    end
    @testset "matricize (eltype=$elt)" for elt in elts
        a = randn(elt, 2, 3, 4, 5)

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
        @test_throws ArgumentError matricize(a, (1, 2), (3,))

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

    @testset "matricizeop (eltype=$elt)" for elt in elts
        rng = StableRNG(123)
        a = randn(rng, elt, 2, 3, 4)

        # identity op: should match matricize exactly
        m = TensorAlgebra.matricizeop(identity, a, (1,), (2, 3))
        m_ref = matricize(a, (1,), (2, 3))
        @test m ≈ m_ref

        m = TensorAlgebra.matricizeop(identity, a, (3, 1), (2,))
        m_ref = matricize(a, (3, 1), (2,))
        @test m ≈ m_ref

        m = TensorAlgebra.matricizeop(identity, a, (2, 3), (1,))
        m_ref = matricize(a, (2, 3), (1,))
        @test m ≈ m_ref

        # conj op
        m = TensorAlgebra.matricizeop(conj, a, (1,), (2, 3))
        m_ref = conj.(matricize(a, (1,), (2, 3)))
        @test m ≈ m_ref

        m = TensorAlgebra.matricizeop(conj, a, (3, 1), (2,))
        m_ref = conj.(matricize(a, (3, 1), (2,)))
        @test m ≈ m_ref
    end

    @testset "unmatricize (eltype=$elt)" for elt in elts
        a0 = randn(elt, 2, 3, 4, 5)
        axes0 = axes(a0)
        m = reshape(a0, 6, 20)

        a = unmatricize(m, axes0[1:2], axes0[3:4])
        @test eltype(a) === elt
        @test a ≈ a0

        a = unmatricize(m, axes0, (1, 2), (3, 4))
        @test eltype(a) === elt
        @test a ≈ a0

        perm_codomain = (4, 2)
        perm_domain = (1, 3)
        invperm_codomain = (3, 2)
        invperm_domain = (4, 1)
        perm = (4, 2, 1, 3)
        a = unmatricize(m, map(i -> axes0[i], perm), invperm_codomain, invperm_domain)
        @test eltype(a) === elt
        @test a ≈ permutedims(a0, perm)

        a = similar(a0)
        unmatricize!(a, m, (1, 2), (3, 4))
        @test a ≈ a0

        m1 = matricize(a0, perm_codomain, perm_domain)
        a = unmatricize(m1, axes0, perm_codomain, perm_domain)
        @test a ≈ a0

        a1 = permutedims(a0, perm)
        a = similar(a1)
        unmatricize!(a, m, invperm_codomain, invperm_domain)
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

        @test_throws ArgumentError unmatricize(m, (), (1, 2), (3,))
        @test_throws ArgumentError unmatricize!(m, m, (1, 2), (3,))
    end

    alg_tensoroperations = ContractAlgorithm(TensorOperations.StridedBLAS())
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
            @test issetequal(labels_dest′, symdiff(labels1, labels2))
            a_dest_tensoroperations, = contract(
                a1, labels1, a2, labels2; alg = alg_tensoroperations
            )
            @test a_dest ≈ a_dest_tensoroperations

            # Specify destination labels
            a_dest = contract(labels_dest, a1, labels1, a2, labels2)
            a_dest_tensoroperations = contract(
                labels_dest, a1, labels1, a2, labels2; alg = alg_tensoroperations
            )
            @test a_dest ≈ a_dest_tensoroperations

            a_dest = contract(labels_dest′, a1, labels1, a2, labels2)
            a_dest_tensoroperations = contract(
                labels_dest′, a1, labels1, a2, labels2; alg = alg_tensoroperations
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
                a_dest_tensoroperations, labels_dest, a1, labels1, a2, labels2, α, β;
                alg = alg_tensoroperations
            )
            ## Here we loosened the tolerance because of some floating point roundoff issue.
            ## with Float32 numbers
            @test a_dest ≈ a_dest_tensoroperations rtol = 50 * default_rtol(elt_dest)
        end
    end
    @testset "integer relabeling (label_type)" begin
        @test TensorAlgebra.label_type((1, 2)) === Int
        @test TensorAlgebra.label_type((:a, :b)) === Symbol
        @test TensorAlgebra.label_type((OptInLabel(1), OptInLabel(2))) === Int

        L = OptInLabel
        a1 = randn(2, 3, 4)
        a2 = randn(3, 4, 5)
        # Shared labels 2 and 3 are contracted; 1 and 4 are the uncontracted, destination labels.
        a_dest, labels_dest = contract(a1, (L(1), L(2), L(3)), a2, (L(2), L(3), L(4)))
        a_ref, = contract(a1, (1, 2, 3), a2, (2, 3, 4))
        @test a_dest ≈ a_ref
        @test labels_dest == [L(1), L(4)]

        # Specifying the destination labels still works for opted-in types.
        a_dest = contract([L(1), L(4)], a1, (L(1), L(2), L(3)), a2, (L(2), L(3), L(4)))
        @test a_dest ≈ a_ref
    end
    @testset "outer product contraction (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts,
            elt2 in elts

        elt_dest = promote_type(elt1, elt2)

        rng = StableRNG(123)
        a1 = randn(rng, elt1, 2, 3)
        a2 = randn(rng, elt2, 4, 5)

        a_dest, labels = contract(a1, ("i", "j"), a2, ("k", "l"))
        @test labels == ["i", "j", "k", "l"]
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
    @testset "contractopadd! (eltype1=$elt1, eltype2=$elt2)" for elt1 in elts, elt2 in elts
        elt_dest = promote_type(elt1, elt2)
        dims = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        labels = (:a, :b, :c, :d, :e, :f, :g, :h, :i)
        rng = StableRNG(123)
        for (d1s, d2s, d_dests) in (
                ((1, 2), (2, 3), (1, 3)),
                ((1, 2), (2, 3), (3, 1)),
                ((1, 2, 3), (2, 3, 4), (1, 4)),
                ((3, 2, 1), (4, 2, 3), (4, 1)),
                ((1, 2, 3), (3, 4), (2, 4, 1)),
            )
            a1 = randn(rng, elt1, map(i -> dims[i], d1s))
            labels1 = map(i -> labels[i], d1s)
            a2 = randn(rng, elt2, map(i -> dims[i], d2s))
            labels2 = map(i -> labels[i], d2s)
            labels_dest = map(i -> labels[i], d_dests)

            α = elt_dest(1.2)
            β = elt_dest(2.4)
            a_dest_init = randn(rng, elt_dest, map(i -> dims[i], d_dests))

            # identity ops should match contractadd!
            a_dest = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest, labels_dest,
                identity, a1, labels1,
                identity, a2, labels2,
                α, β
            )
            a_dest_ref = copy(a_dest_init)
            contractadd!(a_dest_ref, labels_dest, a1, labels1, a2, labels2, α, β)
            @test a_dest ≈ a_dest_ref

            # conj on first input
            a_dest = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest, labels_dest,
                conj, a1, labels1,
                identity, a2, labels2,
                α, β
            )
            a_dest_ref = copy(a_dest_init)
            contractadd!(a_dest_ref, labels_dest, conj.(a1), labels1, a2, labels2, α, β)
            @test a_dest ≈ a_dest_ref

            # compare against TensorOperations backend
            a_dest_to = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest_to, labels_dest,
                conj, a1, labels1,
                identity, a2, labels2,
                α, β; alg = alg_tensoroperations
            )
            @test a_dest ≈ a_dest_to

            # conj on second input
            a_dest = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest, labels_dest,
                identity, a1, labels1,
                conj, a2, labels2,
                α, β
            )
            a_dest_ref = copy(a_dest_init)
            contractadd!(a_dest_ref, labels_dest, a1, labels1, conj.(a2), labels2, α, β)
            @test a_dest ≈ a_dest_ref

            # compare against TensorOperations backend
            a_dest_to = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest_to, labels_dest,
                identity, a1, labels1,
                conj, a2, labels2,
                α, β; alg = alg_tensoroperations
            )
            @test a_dest ≈ a_dest_to

            # conj on both inputs
            a_dest = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest, labels_dest,
                conj, a1, labels1,
                conj, a2, labels2,
                α, β
            )
            a_dest_ref = copy(a_dest_init)
            contractadd!(
                a_dest_ref, labels_dest, conj.(a1), labels1, conj.(a2), labels2, α, β
            )
            @test a_dest ≈ a_dest_ref

            # compare against TensorOperations backend
            a_dest_to = copy(a_dest_init)
            TensorAlgebra.contractopadd!(
                a_dest_to, labels_dest,
                conj, a1, labels1,
                conj, a2, labels2,
                α, β; alg = alg_tensoroperations
            )
            @test a_dest ≈ a_dest_to
        end
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
        @test labels_dest == collect(labels_a)
        @test a_dest ≈ a * s[]

        # Scalar-array contraction.
        a_dest, labels_dest = contract(s, (), a, labels_a)
        @test labels_dest == collect(labels_a)
        @test a_dest ≈ a * s[]

        # Scalar-scalar contraction.
        a_dest, labels_dest = contract(s, (), t, ())
        @test isempty(labels_dest)
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
