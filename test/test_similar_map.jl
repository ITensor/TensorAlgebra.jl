using Random: default_rng
using TensorAlgebra: TensorAlgebra, rand_map, randn_map, similar_map, zeros_map
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

# The flat axis-friendly constructors fill Base's gap: `randn`/`rand` reject `Base.OneTo`.
@testset "flat construction ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    ax = (Base.OneTo(2), Base.OneTo(3))
    z = TensorAlgebra.zeros(T, ax)
    @test z isa Matrix{T}
    @test size(z) == (2, 3)
    @test iszero(z)
    for f in (TensorAlgebra.randn, TensorAlgebra.rand)
        a = f(default_rng(), T, ax)
        @test a isa Matrix{T}
        @test size(a) == (2, 3)
    end
end

# The dense map constructors flatten `(codomain_axes..., conj.(domain_axes)...)`; `conj` is a
# no-op on a dense axis, so the shape is the concatenation of codomain and domain lengths.
@testset "map construction ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    cod = (Base.OneTo(2), Base.OneTo(3))
    dom = (Base.OneTo(4),)

    z = zeros_map(T, cod, dom)
    @test z isa Array{T, 3}
    @test size(z) == (2, 3, 4)
    @test iszero(z)

    for f in (randn_map, rand_map)
        # Fully specified.
        a = f(default_rng(), T, cod, dom)
        @test a isa Array{T, 3}
        @test size(a) == (2, 3, 4)
        # Element type defaults to `Float64`.
        b = f(cod, dom)
        @test b isa Array{Float64, 3}
        @test size(b) == (2, 3, 4)
    end

    # An empty domain (all-codomain map) is how a plain tensor is constructed.
    zc = zeros_map(T, cod, ())
    @test size(zc) == (2, 3)
end
