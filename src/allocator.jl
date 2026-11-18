"""
    TensorAlgebra.DefaultAllocator()

Allocate with `similar`, with memory managed by Julia's garbage collector. The allocator of an
algorithm that carries none.
"""
struct DefaultAllocator end

"""
    TensorAlgebra.allocator(alg)

The allocator `alg` allocates its temporaries with, [`TensorAlgebra.DefaultAllocator`](@ref)
unless `alg` carries one.
"""
allocator(alg) = DefaultAllocator()

"""
    TensorAlgebra.output_allocator(allocator)

The allocator for the output of an operation whose temporaries come from `allocator`.
[`TensorAlgebra.DefaultAllocator`](@ref) unless `allocator` overloads it, since an output
outlives the call and must not use memory that [`TensorAlgebra.allocator_reset!`](@ref) reuses.
"""
output_allocator(allocator) = DefaultAllocator()

"""
    TensorAlgebra.allocator_checkpoint!(allocator) -> checkpoint
    TensorAlgebra.allocator_reset!(allocator, checkpoint)

Record the state of `allocator`, and release the memory it handed out since `checkpoint` was
recorded. Calls nest, and do nothing for an allocator that does not reuse memory.
"""
allocator_checkpoint!(allocator) = nothing
allocator_reset!(allocator, checkpoint) = allocator
@doc (@doc allocator_checkpoint!) allocator_reset!
