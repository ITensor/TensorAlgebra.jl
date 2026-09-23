using LinearAlgebra: I
using TensorAlgebra:
    TensorAlgebra as TA, MatricizeContract, MatricizeStyle, ReshapeMatricize
using Test: @test, @testset

module MatricizeStyleTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct MyArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::A
    end
    struct MyArrayMatricize <: TA.MatricizeStyle end
    TA.MatricizeStyle(::Type{<:MyArray}) = MyArrayMatricize()
    # Minimal hooks so a round trip (`one!`) can run through the custom style. All of them
    # dispatch on `MyArrayMatricize`, so a path whose style was re-derived from the plain fused
    # matrix instead of threaded through would miss them and error.
    function TA.is_output_view(
            ::typeof(TA.matricizeop), ::MyArrayMatricize, op, a, perm_codomain, perm_domain
        )
        return false
    end
    function TA.allocate_output(
            ::typeof(TA.matricizeop), ::MyArrayMatricize, op, a::MyArray,
            perm_codomain, perm_domain
        )
        return TA.allocate_output(
            TA.matricizeop, TA.ReshapeMatricize(), op, a.parent, perm_codomain, perm_domain
        )
    end
    function TA.matricizeop!(
            dest, ::MyArrayMatricize, op, a::MyArray, perm_codomain, perm_domain
        )
        return TA.matricizeop!(
            dest, TA.ReshapeMatricize(), op, a.parent, perm_codomain, perm_domain
        )
    end
    function TA.unmatricize!(
            ::MyArrayMatricize, a_dest::MyArray, m,
            perm_codomain, perm_domain
        )
        TA.unmatricize!(
            TA.ReshapeMatricize(), a_dest.parent, m, perm_codomain, perm_domain
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
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a1), typeof(a1), typeof(a1)}) ≡
        MatricizeContract(ReshapeMatricize())
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a1), typeof(a1), typeof(a2)}) ≡
        MatricizeContract(ReshapeMatricize())
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a2), typeof(a2), typeof(a1)}) ≡
        MatricizeContract(ReshapeMatricize())
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a2), typeof(a2), typeof(a2)}) ≡
        MatricizeContract(MyArrayMatricize())
end

@testset "style threads through the unfold" begin
    # `one!` folds with the caller-supplied style and must unfold with the same style, not one
    # re-derived from the fused matrix (here a plain `Matrix`, whose derived style would be
    # `ReshapeMatricize` and would not know how to scatter into a `MyArray`).
    A = MyArray(randn(3, 3))
    TA.one!(MyArrayMatricize(), A, Val(1))
    @test A.parent ≈ I
end
