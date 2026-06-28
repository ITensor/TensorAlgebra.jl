using Adapt: adapt
using JLArrays: JLArray
using TensorAlgebra:
    ConjArray, add!, bipermutedimsopadd!, conjed, permutedimsadd!, permutedimsopadd!
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
    @testset "bipermutedimsopadd! unwraps PermutedDimsArray src (arraytype=$arrayt)" for arrayt in
        (
            Array,
            JLArray,
        )
        dev = adapt(arrayt)
        parent = dev(randn(2, 3, 4, 5))
        w = (3, 1, 4, 2)
        src = PermutedDimsArray(parent, w)
        for (pc, pd) in (((1, 2, 3, 4), ()), ((2, 4), (1, 3)), ((3, 1), (2, 4)))
            perm = (pc..., pd...)
            ref = permutedims(permutedims(parent, w), perm)
            for β in (0, 3)
                dest = dev(randn(size(ref)...))
                dest′ = copy(dest)
                bipermutedimsopadd!(dest′, identity, src, pc, pd, 2, β)
                @test dest′ ≈ β * dest + 2 * ref
            end
        end
    end
    @testset "bipermutedimsopadd! unwraps ConjArray src (arraytype=$arrayt)" for arrayt in
        (
            Array,
            JLArray,
        )
        dev = adapt(arrayt)
        parent = dev(randn(ComplexF64, 2, 3, 4, 5))
        src = ConjArray(parent)
        for (pc, pd) in (((1, 2, 3, 4), ()), ((2, 4), (1, 3)), ((3, 1), (2, 4)))
            perm = (pc..., pd...)
            # `op = identity` composes with the wrapper's `conj`: result is the conjugated,
            # permuted parent.
            ref_conj = permutedims(conj(parent), perm)
            # `op = conj` cancels the wrapper's `conj`: result is the bare permuted parent.
            ref_id = permutedims(parent, perm)
            for β in (0, 3)
                dest = dev(randn(ComplexF64, size(ref_conj)...))
                dest′ = copy(dest)
                bipermutedimsopadd!(dest′, identity, src, pc, pd, 2, β)
                @test dest′ ≈ β * dest + 2 * ref_conj

                dest′ = copy(dest)
                bipermutedimsopadd!(dest′, conj, src, pc, pd, 2, β)
                @test dest′ ≈ β * dest + 2 * ref_id
            end
        end
    end
    @testset "add!(b, conjed(a)) matches eager conj (arraytype=$arrayt)" for arrayt in
        (
            Array,
            JLArray,
        )
        dev = adapt(arrayt)
        a = dev(randn(ComplexF64, 2, 3, 4))
        α = 2
        for β in (0, 3)
            b = dev(randn(ComplexF64, 2, 3, 4))
            b_lazy = copy(b)
            b_eager = copy(b)
            add!(b_lazy, conjed(a), α, β)
            add!(b_eager, conj(a), α, β)
            @test b_lazy ≈ b_eager
        end
    end
    @testset "bipermutedimsopadd! 0-dim with β=0 must not read dest (eltype=$T)" for T in
        (
            Float64,
            BigFloat,
        )
        # With β=0, `dest` is write-only by BLAS convention; its contents need not be
        # defined. For element types whose `undef` storage is unreadable (e.g. mutable
        # `BigFloat`), reading the slot would throw `UndefRefError`.
        src = fill(T(7))
        for op in (identity, conj)
            dest = Array{T, 0}(undef)
            bipermutedimsopadd!(dest, op, src, (), (), true, false)
            @test dest[] == op(src[])
        end
        # With β nonzero, both reads and writes go through with the accumulating
        # semantics `dest = β * dest + α * op(src)`.
        dest = fill(T(2))
        bipermutedimsopadd!(dest, identity, src, (), (), T(3), T(5))
        @test dest[] == 3 * 7 + 5 * 2
    end
    @testset "permutedimsopadd! (arraytype=$arrayt)" for arrayt in (Array,)
        dev = adapt(arrayt)
        a = dev(randn(ComplexF64, 2, 2, 2))
        perm = (3, 1, 2)
        α = 2
        for β in (0, 3)
            b = dev(randn(ComplexF64, 2, 2, 2))
            b′ = copy(b)
            permutedimsopadd!(b′, conj, a, perm, α, β)
            @test b′ ≈ β * b + α * permutedims(conj(a), perm)
        end
        # identity op should match permutedimsadd!
        for β in (0, 3)
            b = dev(randn(ComplexF64, 2, 2, 2))
            b′ = copy(b)
            b″ = copy(b)
            permutedimsopadd!(b′, identity, a, perm, α, β)
            permutedimsadd!(b″, a, perm, α, β)
            @test b′ ≈ b″
        end
    end
end
