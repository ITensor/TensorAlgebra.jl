using LinearAlgebra: I
using MatrixAlgebraKit: truncrank
using TensorAlgebra: TensorAlgebra as TA, MatricizeContract
using TensorOperations: BufferAllocator
using Test: @test, @test_throws, @testset

module AlgorithmsTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    # Counts its calls and permutes like the default algorithm.
    struct CountingPermute <: TA.PermuteAlgorithm
        count::Base.RefValue{Int}
    end
    CountingPermute() = CountingPermute(Ref(0))
    function TA.bipermutedimsconjadd!(
            alg::CountingPermute, dest::Array, isconj::Bool, src::Array,
            perm_codomain, perm_domain, α::Number, β::Number
        )
        alg.count[] += 1
        return TA.bipermutedimsconjadd!(
            TA.DefaultPermuteAlgorithm(), dest, isconj, src, perm_codomain, perm_domain, α, β
        )
    end
    struct UnimplementedPermute <: TA.PermuteAlgorithm end

    # Counts the dense allocations it serves and the checkpoints still open.
    mutable struct CountingAllocator
        count::Int
        depth::Int
    end
    CountingAllocator() = CountingAllocator(0, 0)
    function TA.similar_map(
            allocator::CountingAllocator, prototype::Array, ::Type{T},
            axes_codomain::Tuple{Vararg{Base.OneTo}}, axes_domain::Tuple{Vararg{Base.OneTo}}
        ) where {T}
        allocator.count += 1
        return TA.similar_map(prototype, T, axes_codomain, axes_domain)
    end
    TA.allocator_checkpoint!(allocator::CountingAllocator) = (allocator.depth += 1)
    function TA.allocator_reset!(allocator::CountingAllocator, checkpoint)
        allocator.depth -= 1
        return allocator
    end
end
using .AlgorithmsTestUtils: CountingAllocator, CountingPermute, UnimplementedPermute

@testset "permute algorithm" begin
    a = randn(3, 4, 5)
    @test TA.default_algorithm(TA.bipermutedimsconjadd!, (a, a)) ≡
        TA.DefaultPermuteAlgorithm()
    @test TA.select_algorithm(
        TA.bipermutedimsconjadd!, TA.DefaultPermuteAlgorithm(), (a, a)
    ) ≡ TA.DefaultPermuteAlgorithm()
    @test TA.permutedims(a, (3, 1, 2); alg = TA.DefaultPermuteAlgorithm()) ==
        permutedims(a, (3, 1, 2))

    alg = CountingPermute()
    @test TA.permutedims(a, (3, 1, 2); alg) == permutedims(a, (3, 1, 2))
    @test alg.count[] == 1
    @test TA.matricize(a, (2,), (3, 1); alg) == reshape(permutedims(a, (2, 3, 1)), 4, 15)
    @test alg.count[] == 2
    # A view does not permute.
    TA.matricize(a, (1,), (2, 3); alg)
    @test alg.count[] == 2

    b = randn(5, 2, 4)
    c, = TA.contract(a, (:i, :j, :k), b, (:k, :l, :j))
    c_alg = TA.contractalign(
        (:l, :i), a, (:i, :j, :k), b, (:k, :l, :j);
        alg = MatricizeContract(; permute_alg = alg)
    )
    @test c_alg ≈ permutedims(c, (2, 1))
    @test alg.count[] > 2

    @test_throws ArgumentError TA.permutedims(a, (3, 1, 2); alg = 1)
    @test_throws MethodError TA.permutedims(a, (3, 1, 2); alg = UnimplementedPermute())
end

@testset "allocator" begin
    @test TA.allocator(MatricizeContract()) ≡ TA.DefaultAllocator()
    @test TA.allocator(TA.DefaultPermuteAlgorithm()) ≡ TA.DefaultAllocator()

    a = randn(3, 4, 5)
    b = randn(5, 2, 4)
    c, = TA.contract(a, (:i, :j, :k), b, (:k, :l, :j))

    # Both operands are contracted over their middle dimension, so both matricizations copy, as
    # does the product, which is permuted into the destination.
    a2 = randn(2, 4, 6)
    c2, = TA.contract(a, (:i, :j, :k), a2, (:l, :j, :m))
    allocator = CountingAllocator()
    c2_alg = TA.contractalign(
        (:m, :i, :k, :l), a, (:i, :j, :k), a2, (:l, :j, :m);
        alg = MatricizeContract(; allocator)
    )
    @test c2_alg ≈ permutedims(c2, (4, 1, 2, 3))
    @test allocator.count == 3
    @test allocator.depth == 0

    buffer = BufferAllocator()
    alg = MatricizeContract(; allocator = buffer)
    for _ in 1:2
        c2_alg = TA.contractalign((:m, :i, :k, :l), a, (:i, :j, :k), a2, (:l, :j, :m); alg)
        @test c2_alg ≈ permutedims(c2, (4, 1, 2, 3))
        @test buffer.offset == 0
    end
    @test buffer.max_offset > 0

    # A temporary is served from the buffer until the buffer is reset.
    checkpoint = TA.allocator_checkpoint!(buffer)
    m = TA.allocate_output(TA.matricizeconj, false, a, (2,), (3, 1); allocator = buffer)
    @test size(m) == (4, 15)
    @test UInt(pointer(m)) == UInt(pointer(buffer.buffer))
    @test buffer.offset == sizeof(m)
    TA.allocator_reset!(buffer, checkpoint)
    @test buffer.offset == 0
end

@testset "factorizations with a permute algorithm and an allocator" begin
    # A Hermitian positive definite map from dimensions `(3, 2)` to `(1, 4)`, so every
    # factorization applies and the matricization permutes.
    perm_codomain, perm_domain = (3, 2), (1, 4)
    m = randn(6, 6)
    m = m * m' + I
    a = permutedims(reshape(m, 2, 3, 2, 3), invperm((perm_codomain..., perm_domain...)))
    trunc = truncrank(3)
    fs_kwargs = (
        (TA.qr_compact, (;)), (TA.qr_full, (;)), (TA.lq_compact, (;)),
        (TA.lq_full, (;)),
        (TA.left_polar, (;)), (TA.right_polar, (;)),
        (TA.left_orth, (;)), (TA.right_orth, (;)),
        (TA.svd_compact, (;)), (TA.svd_full, (;)), (TA.svd_trunc, (; trunc)),
        (TA.svd_vals, (;)),
        (TA.eigh_full, (;)), (TA.eig_full, (;)), (TA.eigh_trunc, (; trunc)),
        (TA.eig_trunc, (; trunc)), (TA.eigh_vals, (;)), (TA.eig_vals, (;)),
        (TA.left_null, (;)), (TA.right_null, (;)), (TA.project_hermitian, (;)),
        (TA.sqrth_safe, (;)), (TA.invsqrth_safe, (;)), (TA.sqrth_invsqrth_safe, (;)),
    )
    @testset "$f" for (f, kwargs) in fs_kwargs
        F = f(a, perm_codomain, perm_domain; kwargs...)
        F = F isa Tuple ? F : (F,)

        permute_alg = CountingPermute()
        F_alg = f(a, perm_codomain, perm_domain; permute_alg, kwargs...)
        @test all(map(≈, F_alg isa Tuple ? F_alg : (F_alg,), F))
        @test permute_alg.count[] == 1

        buffer = BufferAllocator()
        # The first call sizes the buffer, the second is served from it.
        for _ in 1:2
            F_buffer = f(a, perm_codomain, perm_domain; allocator = buffer, kwargs...)
            F_buffer = F_buffer isa Tuple ? F_buffer : (F_buffer,)
            @test all(map(≈, F_buffer, F))
            @test buffer.offset == 0
        end
        # `project_hermitian` returns its input, so it does not take it from the allocator.
        @test (buffer.max_offset > 0) == (f !== TA.project_hermitian)
        F_buffer = f(a, perm_codomain, perm_domain; allocator = buffer, kwargs...)
        F_buffer = F_buffer isa Tuple ? F_buffer : (F_buffer,)
        # Overwrite the released matricized input: no factor may share memory with it.
        checkpoint = TA.allocator_checkpoint!(buffer)
        a_mat = TA.allocate_output(
            TA.matricizeconj, false, a, perm_codomain, perm_domain; allocator = buffer
        )
        fill!(parent(a_mat), NaN)
        TA.allocator_reset!(buffer, checkpoint)
        @test all(map(≈, F_buffer, F))
    end
end
