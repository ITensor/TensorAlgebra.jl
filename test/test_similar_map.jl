using TensorAlgebra: similar_map
using Test: @test, @testset

@testset "similar_map ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    prototype = randn(T, 3)
    cod = (Base.OneTo(2), Base.OneTo(3))
    dom = (Base.OneTo(4), Base.OneTo(5))

    # With explicit element type.
    O = similar_map(prototype, Float32, cod, dom)
    @test eltype(O) === Float32
    @test size(O) == (2, 3, 4, 5)

    # Element type defaults to `eltype(prototype)`.
    O2 = similar_map(prototype, cod, dom)
    @test eltype(O2) === T
    @test size(O2) == (2, 3, 4, 5)

    # Mixed-eltype prototype propagates to the default.
    O3 = similar_map(zeros(ComplexF32, 1), (Base.OneTo(2),), (Base.OneTo(3),))
    @test eltype(O3) === ComplexF32
    @test size(O3) == (2, 3)
end
