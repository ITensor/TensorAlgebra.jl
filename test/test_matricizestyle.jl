using TensorAlgebra: TensorAlgebra as TA, Matricize, MatricizeStyle, ReshapeMatricize
using Test: @test, @testset

module MatricizeStyleTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct MyArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::A
    end
    struct MyArrayMatricize <: TA.MatricizeStyle end
    TA.MatricizeStyle(::Type{<:MyArray}) = MyArrayMatricize()
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
