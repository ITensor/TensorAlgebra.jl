using TensorAlgebra: TensorAlgebra as TA, FusionStyle, Matricize, ReshapeFusion
using Test: @test, @testset

module FusionStyleTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct MyArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::A
    end
    struct MyArrayFusion <: TA.FusionStyle end
    TA.FusionStyle(::Type{<:MyArray}) = MyArrayFusion()
end
using .FusionStyleTestUtils: MyArray, MyArrayFusion

@testset "FusionStyle" begin
    a1 = randn(2, 2)
    a2 = MyArray(randn(2, 2))
    @test FusionStyle(a1) ≡ ReshapeFusion()
    @test FusionStyle(a2) ≡ MyArrayFusion()
    @test FusionStyle(typeof(a1)) ≡ ReshapeFusion()
    @test FusionStyle(ReshapeFusion(), ReshapeFusion()) ≡ ReshapeFusion()
    @test FusionStyle(MyArrayFusion(), MyArrayFusion()) ≡ MyArrayFusion()
    @test FusionStyle(MyArrayFusion(), ReshapeFusion()) ≡ ReshapeFusion()
    @test FusionStyle(ReshapeFusion(), MyArrayFusion()) ≡ ReshapeFusion()
    @test TA.default_contract_algorithm(typeof(a1), typeof(a1)) ≡ Matricize(ReshapeFusion())
    @test TA.default_contract_algorithm(typeof(a1), typeof(a2)) ≡ Matricize(ReshapeFusion())
    @test TA.default_contract_algorithm(typeof(a2), typeof(a1)) ≡ Matricize(ReshapeFusion())
    @test TA.default_contract_algorithm(typeof(a2), typeof(a2)) ≡ Matricize(MyArrayFusion())
end
