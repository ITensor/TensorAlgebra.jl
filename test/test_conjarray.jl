using LinearAlgebra: Diagonal, norm
using TensorAlgebra: TensorAlgebra as TA, ConjArray, bipermutedimsopadd!, conjed, contract,
    contractopadd!, matricize, permutedims, qr_compact
using Test: @test, @test_throws, @testset

@testset "ConjArray (plain arrays)" begin
    a = randn(ComplexF64, 3, 4)
    c = conjed(a)
    @test c isa ConjArray
    @test parent(c) === a
    @test conjed(c) === a
    @test eltype(c) === eltype(a)
    @test size(c) == size(a)
    @test axes(c) == axes(a)
    @test copy(c) == conj(a)
    @test conj(c) === a
end

@testset "ConjArray goes through its parent instead of indexing" begin
    a = randn(ComplexF64, 3, 4)
    b = randn(ComplexF64, 3, 4)
    c = conjed(a)
    @test_throws ArgumentError c[2, 3]
    @test_throws ArgumentError c[2, 3] = 1
    @test collect(c) == conj(a)

    @test c == conj(a)
    @test conj(a) == c
    @test c == conjed(copy(a))
    @test c ≈ conj(a)
    @test norm(c) == norm(a)
    @test Array(c) == conj(a)
    @test sum(c) ≈ sum(conj(a))
    @test maximum(real, c) == maximum(real, a)
    @test Base.permutedims(c, (2, 1)) == permutedims(conj(a), (2, 1))
    @test c .+ b == conj(a) .+ b
    @test abs.(c) == abs.(a)
    @test sprint(show, c) == "conjed($(sprint(show, a)))"
    @test startswith(sprint(show, MIME"text/plain"(), c), "conjed of ")

    # Writes go to the parent as the conjugate of what is written.
    d = conjed(zeros(ComplexF64, 3, 4))
    @test copyto!(d, b) === d
    @test parent(d) == conj(b)
    @test copyto!(similar(b), d) == b
    @test fill!(d, 2im) === d
    @test parent(d) == fill(-2im, 3, 4)
    @test TA.zero!(d) === d
    @test iszero(parent(d))
    @test similar(c, Float64, (2, 5)) isa Matrix{Float64}
end

@testset "factorizations of a ConjArray" begin
    a = randn(ComplexF64, 2, 3, 4)
    Q, R = qr_compact(conjed(a), (1, 2), (3,))
    Qe, Re = qr_compact(conj(a), (1, 2), (3,))
    @test Q ≈ Qe
    @test R ≈ Re
end

@testset "ConjArray is absorbed by permutation and matricize" begin
    a = randn(ComplexF64, 2, 3, 4, 5)
    c = conjed(a)
    @test permutedims(c, (3, 1, 4, 2)) == permutedims(conj(a), (3, 1, 4, 2))
    for (pc, pd) in (((1, 2, 3, 4), ()), ((2, 4), (1, 3)))
        dest = zeros(ComplexF64, size(a)[[pc..., pd...]]...)
        bipermutedimsopadd!(dest, identity, c, pc, pd, true, false)
        @test dest == permutedims(conj(a), (pc..., pd...))
        dest = zeros(ComplexF64, size(a)[[pc..., pd...]]...)
        bipermutedimsopadd!(dest, conj, c, pc, pd, true, false)
        @test dest == permutedims(a, (pc..., pd...))
    end
    @test matricize(c, (2, 4), (1, 3)) == matricize(conj(a), (2, 4), (1, 3))
    @test matricize(c, Val(2)) == matricize(conj(a), Val(2))
end

@testset "ConjArray is absorbed by contraction" begin
    a = randn(ComplexF64, 2, 3, 4)
    b = randn(ComplexF64, 4, 3, 5)
    labels_a, labels_b = (:i, :j, :k), (:k, :j, :l)
    for (x, y) in ((conjed(a), b), (a, conjed(b)), (conjed(a), conjed(b)))
        xe = x isa ConjArray ? copy(x) : x
        ye = y isa ConjArray ? copy(y) : y
        r, labels_r = contract(x, labels_a, y, labels_b)
        re, labels_re = contract(xe, labels_a, ye, labels_b)
        @test labels_r == labels_re
        @test r == re
    end

    # The conjugate is folded into the op passed alongside it.
    dest = zeros(ComplexF64, 2, 5)
    contractopadd!(
        dest,
        (:i, :l),
        conj,
        conjed(a),
        labels_a,
        identity,
        b,
        labels_b,
        true,
        false
    )
    @test dest == first(contract(a, labels_a, b, labels_b))
end

@testset "contractpermop allocates for the unwrapped operands" begin
    a = randn(ComplexF64, 2, 3)
    b = randn(ComplexF64, 3, 4)
    @test TA.contractpermop((1,), (2,), conj, a, (1,), (2,), identity, b, (1,), (2,)) ≈
        conj(a) * b
    @test TA.contractpermop(
        (1,),
        (2,),
        identity,
        conjed(a),
        (1,),
        (2,),
        identity,
        b,
        (1,),
        (2,)
    ) ≈
        conj(a) * b

    # Allocation sees the parent, so a `Diagonal` pair still allocates a `Diagonal`.
    d1 = Diagonal(randn(ComplexF64, 3))
    d2 = Diagonal(randn(ComplexF64, 3))
    r, _ = contract(conjed(d1), (:i, :j), d2, (:j, :k))
    @test r isa Diagonal
    @test r ≈ conj(d1) * d2
end

@testset "op-aware contraction check_input" begin
    a = randn(ComplexF64, 2, 3)
    b = randn(ComplexF64, 3, 4)
    @test isnothing(
        TA.check_input(
            TA.contractpermopadd!,
            zeros(ComplexF64, 2, 4),
            (1,),
            (2,),
            conj,
            a,
            (1,),
            (2,),
            identity,
            b,
            (1,),
            (2,)
        )
    )
    @test_throws ArgumentError TA.check_input(
        TA.contractpermopadd!, zeros(ComplexF64, 2, 4), (1,), (2,),
        abs, a, (1,), (2,), identity, b, (1,), (2,)
    )
    @test_throws ArgumentError TA.check_input(
        TA.contractpermopadd!, zeros(ComplexF64, 2, 4), (1,), (2,),
        conj, a, (1,), (2,), identity, randn(4, 4), (1,), (2,)
    )
end
