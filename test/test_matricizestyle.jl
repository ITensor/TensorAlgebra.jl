using LinearAlgebra: I
using TensorAlgebra: TensorAlgebra as TA, Matricize, MatricizeStyle, ReshapeMatricize
using Test: @test, @testset

module MatricizeStyleTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct MyArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::A
    end
    struct MyArrayMatricize <: TA.MatricizeStyle end
    TA.MatricizeStyle(::Type{<:MyArray}) = MyArrayMatricize()
    # Minimal fold/unfold leaves so a round-trip (`one!`) can run through the custom style:
    # both dispatch on `MyArrayMatricize`, so an unfold whose style was re-derived from the
    # plain fused matrix instead of threaded through would miss them and error.
    TA.ismatricizeview(::MyArrayMatricize, a, ::Val) = false
    function TA.matricizecopy(::MyArrayMatricize, a::MyArray, ndims_codomain::Val)
        return TA.matricizecopy(TA.ReshapeMatricize(), a.parent, ndims_codomain)
    end
    function TA.unmatricizeperm!(
            ::MyArrayMatricize, a_dest::MyArray, m,
            invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
        )
        TA.unmatricizeperm!(
            TA.ReshapeMatricize(), a_dest.parent, m, invperm_codomain, invperm_domain
        )
        return a_dest
    end
end
using .MatricizeStyleTestUtils: MyArray, MyArrayMatricize

@testset "MatricizeStyle" begin
    a1 = randn(2, 2)
    a2 = MyArray(randn(2, 2))
    @test MatricizeStyle(a1) ≡ ReshapeMatricize()
    @test MatricizeStyle(a2) ≡ MyArrayMatricize()
    @test MatricizeStyle(typeof(a1)) ≡ ReshapeMatricize()
    @test MatricizeStyle(ReshapeMatricize(), ReshapeMatricize()) ≡ ReshapeMatricize()
    @test MatricizeStyle(MyArrayMatricize(), MyArrayMatricize()) ≡ MyArrayMatricize()
    @test MatricizeStyle(MyArrayMatricize(), ReshapeMatricize()) ≡ ReshapeMatricize()
    @test MatricizeStyle(ReshapeMatricize(), MyArrayMatricize()) ≡ ReshapeMatricize()
    @test TA.default_contract_algorithm(typeof(a1), typeof(a1)) ≡
        Matricize(ReshapeMatricize())
    @test TA.default_contract_algorithm(typeof(a1), typeof(a2)) ≡
        Matricize(ReshapeMatricize())
    @test TA.default_contract_algorithm(typeof(a2), typeof(a1)) ≡
        Matricize(ReshapeMatricize())
    @test TA.default_contract_algorithm(typeof(a2), typeof(a2)) ≡
        Matricize(MyArrayMatricize())
end

@testset "style threads through the unfold" begin
    # `one!` folds with the caller-supplied style and must unfold with the same style, not one
    # re-derived from the fused matrix (here a plain `Matrix`, whose derived style would be
    # `ReshapeMatricize` and would not know how to scatter into a `MyArray`).
    A = MyArray(randn(3, 3))
    TA.one!(MyArrayMatricize(), A, Val(1))
    @test A.parent ≈ I
end
