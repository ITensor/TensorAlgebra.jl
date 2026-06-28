using TensorAlgebra: TensorAlgebra as TA, ConjArray, bipermutedimsopadd!, conjed
using Test: @test, @testset

@testset "ConjArray (plain arrays)" begin
    a = randn(ComplexF64, 3, 4)

    c = conjed(a)
    @test c isa ConjArray
    @test parent(c) === a
    @test conjed(c) === a            # involution unwraps, no nesting
    @test eltype(c) === eltype(a)
    @test ndims(c) == 2
    @test size(c) == size(a)
    @test axes(c) == axes(a)         # plain axes are unchanged by conj

    # Indexing and materialization match eager conj of the parent.
    @test c[2, 3] == conj(a[2, 3])
    @test collect(c) == conj(a)

    # Real eltype: conj is a value no-op, but the wrapper still wraps.
    r = randn(Float64, 2, 2)
    cr = conjed(r)
    @test cr isa ConjArray
    @test collect(cr) == r
    @test axes(cr) == axes(r)
end

@testset "ConjArray composes with PermutedDimsArray" begin
    a = randn(ComplexF64, 2, 3, 4, 5)
    w = (3, 1, 4, 2)

    # ConjArray outside, PermutedDimsArray inside, and vice versa.
    c_out = ConjArray(PermutedDimsArray(a, w))
    c_in = PermutedDimsArray(ConjArray(a), w)
    @test collect(c_out) == conj(permutedims(a, w))
    @test copy(c_out) ≈ conj(permutedims(a, w))
    @test collect(c_in) == permutedims(conj(a), w)

    # Both unwrap through bipermutedimsopadd! (the nested wrappers fold into op and perm).
    for (pc, pd) in (((1, 2, 3, 4), ()), ((2, 4), (1, 3)))
        perm = (pc..., pd...)
        ref_out = permutedims(conj(permutedims(a, w)), perm)
        dest = zeros(ComplexF64, size(ref_out)...)
        bipermutedimsopadd!(dest, identity, c_out, pc, pd, true, false)
        @test dest ≈ ref_out

        ref_in = permutedims(permutedims(conj(a), w), perm)
        dest = zeros(ComplexF64, size(ref_in)...)
        bipermutedimsopadd!(dest, identity, c_in, pc, pd, true, false)
        @test dest ≈ ref_in
    end
end
