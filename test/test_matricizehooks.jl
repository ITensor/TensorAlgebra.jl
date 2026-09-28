using LinearAlgebra: I
using TensorAlgebra: TensorAlgebra as TA, MatricizeContract
using Test: @test, @testset

module MatricizeHooksTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct MyArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::A
    end
    # Minimal hooks so a round trip (`one!`) can run through the wrapper. All of them dispatch on
    # `MyArray`, so a path that dropped the wrapper and re-derived the hooks from the plain fused
    # matrix would miss them and error.
    function TA.is_output_view(
            ::typeof(TA.matricizeop), op, a::MyArray, perm_codomain, perm_domain
        )
        return false
    end
    function TA.allocate_output(
            ::typeof(TA.matricizeop), op, a::MyArray, perm_codomain, perm_domain
        )
        return TA.allocate_output(TA.matricizeop, op, a.parent, perm_codomain, perm_domain)
    end
    function TA.matricizeop!(dest, op, a::MyArray, perm_codomain, perm_domain)
        return TA.matricizeop!(dest, op, a.parent, perm_codomain, perm_domain)
    end
    function TA.unmatricizeadd!(
            a_dest::MyArray, m, perm_codomain, perm_domain, α::Number, β::Number
        )
        TA.unmatricizeadd!(a_dest.parent, m, perm_codomain, perm_domain, α, β)
        return a_dest
    end
end
using .MatricizeHooksTestUtils: MyArray

@testset "MatricizeContract is the dense default" begin
    a1 = randn(2, 2)
    a2 = MyArray(randn(2, 2))
    @test MatricizeContract() ≡ MatricizeContract()
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a1), typeof(a1), typeof(a1)}) ≡
        MatricizeContract()
    @test TA.default_algorithm(TA.contract!, Tuple{typeof(a2), typeof(a2), typeof(a2)}) ≡
        MatricizeContract()
end

@testset "the hooks thread through the unfold" begin
    # `one!` folds through the wrapper's hooks and must unfold through them too, not through
    # hooks re-derived from the fused matrix (here a plain `Matrix`, whose dense `unmatricize!`
    # would not know how to scatter into a `MyArray`).
    A = MyArray(randn(3, 3))
    TA.one!(A, Val(1))
    @test A.parent ≈ I
end
