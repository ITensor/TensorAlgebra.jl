using Adapt: adapt
using JLArrays: JLArray
using TensorAlgebra: add!, permutedimsadd!
using Test: @test, @testset

@testset "[permutedims]add!" begin
    @testset "add!(b, a, α, β) (arraytype=$arrayt)" for arrayt in (Array, JLArray)
        dev = adapt(arrayt)
        a = dev(randn(2, 2, 2))
        α = 2
        for β in (0, 3)
            b = dev(randn(2, 2, 2))
            b′ = copy(b)
            add!(b′, a, α, β)
            @test b′ ≈ β * b + α * a
        end
    end
    @testset "add!(b, a::PermutedDimsArray, α, β) (arraytype=$arrayt)" for arrayt in
        (Array, JLArray)
        dev = adapt(arrayt)
        a = dev(randn(2, 2, 2))
        α = 2
        for β in (0, 3)
            b = dev(randn(2, 2, 2))
            b′ = copy(b)
            add!(b′, PermutedDimsArray(a, (3, 1, 2)), α, β)
            @test b′ ≈ β * b + α * permutedims(a, (3, 1, 2))
        end
    end
    @testset "add!(b, a) (arraytype=$arrayt)" for arrayt in (Array, JLArray)
        dev = adapt(arrayt)
        a = dev(randn(2, 2, 2))
        b = dev(randn(2, 2, 2))
        b′ = copy(b)
        add!(b′, a)
        @test b′ ≈ b + a
    end
    @testset "permutedimsadd! (arraytype=$arrayt)" for arrayt in (Array, JLArray)
        dev = adapt(arrayt)
        a = dev(randn(2, 2, 2))
        perm = (3, 1, 2)
        α = 2
        for β in (0, 3)
            b = dev(randn(2, 2, 2))
            b′ = copy(b)
            permutedimsadd!(b′, a, perm, α, β)
            @test b′ ≈ β * b + α * permutedims(a, perm)
        end
    end
end
