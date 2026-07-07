using TensorAlgebra: TensorAlgebra as TA, ConjBroadcasted, PermutedDims,
    bipermutedimsopadd!, linearbroadcasted
using Test: @test, @testset

@testset "ConjBroadcasted (plain arrays)" begin
    a = randn(ComplexF64, 3, 4)

    c = ConjBroadcasted(a)
    @test parent(c) === a
    @test eltype(c) === eltype(a)
    @test ndims(c) == 2
    @test axes(c) == axes(a)         # plain axes are unchanged by conj

    # The `conj` lowering produces a `ConjBroadcasted`, and a nested conjugate cancels.
    @test linearbroadcasted(conj, a) ≡ c
    @test linearbroadcasted(conj, c) === a

    # Materialization matches eager conj of the parent (via the `LinearBroadcasted` protocol;
    # `ConjBroadcasted` is not an `AbstractArray`, so it is not indexed or `collect`ed).
    @test copy(c) == conj(a)

    # Real eltype: conj is a value no-op, but the wrapper still wraps.
    r = randn(Float64, 2, 2)
    cr = ConjBroadcasted(r)
    @test copy(cr) == r
    @test axes(cr) == axes(r)
end

@testset "ConjBroadcasted composes with permutation wrappers" begin
    a = randn(ComplexF64, 2, 3, 4, 5)
    w = (3, 1, 4, 2)

    # conj outside permute wraps a Base `PermutedDimsArray`; permute outside conj uses the
    # generic `PermutedDims` node, since `ConjBroadcasted` is not an `AbstractArray` and so
    # cannot be a `PermutedDimsArray` parent.
    c_out = ConjBroadcasted(PermutedDimsArray(a, w))
    c_in = PermutedDims(ConjBroadcasted(a), w)
    @test copy(c_out) ≈ conj(permutedims(a, w))

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
