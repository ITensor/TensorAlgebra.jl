using LinearAlgebra: Diagonal

# =======================================  misc  ========================================

"""
    TensorAlgebra.trivialrange(R::Type{<:AbstractUnitRange}[, n::Integer])
    TensorAlgebra.trivialrange(r::AbstractUnitRange[, n::Integer])

Return the identity range for fusing ranges of type `R`: a one-dimensional range
`t` for which fusing `t` with any other range of the same family leaves that range
unchanged. Defaults to `Base.OneTo(1)`, and for a graded range it is a charge-0
one-dimensional sector.

With a length `n`, return the `n`-dimensional analogue: `n` copies of the identity
range stacked into one range (for a graded range, a charge-0 sector of dimension `n`).
"""
trivialrange(r::AbstractUnitRange) = trivialrange(typeof(r))
trivialrange(::Type{<:AbstractUnitRange}) = Base.OneTo(1)
trivialrange(r::AbstractUnitRange, n::Integer) = trivialrange(typeof(r), n)
trivialrange(::Type{<:AbstractUnitRange}, n::Integer) = Base.OneTo(n)

"""
    bipermutedimsop(op, src, perm_codomain, perm_domain; alg = nothing) -> dest

Return `op.(permutedims(src, (perm_codomain..., perm_domain...)))` in new storage, split into
the codomain `perm_codomain` and the domain `perm_domain`. `op` is `identity` or `conj`.

See also [`bipermutedimsconj`](@ref).
"""
function bipermutedimsop(op, src, perm_codomain, perm_domain; kwargs...)
    return bipermutedimsconj(_isconj(op), src, perm_codomain, perm_domain; kwargs...)
end
"""
    bipermutedimsconj(isconj::Bool, src, perm_codomain, perm_domain; alg = nothing,
                      allocator = TensorAlgebra.DefaultAllocator()) -> dest

Return `permutedims(src, (perm_codomain..., perm_domain...))` in new storage, split into the
codomain `perm_codomain` and the domain `perm_domain`, with `src` conjugated when `isconj` is
`true`. `alg` is the [`PermuteAlgorithm`](@ref) and `allocator` provides the new storage. Array
types customize the output by overloading
`allocate_output(bipermutedimsconj, isconj, src, perm_codomain, perm_domain; allocator)`.

See also [`bipermutedimsop`](@ref).
"""
function bipermutedimsconj(isconj::Bool, src, perm_codomain, perm_domain; kwargs...)
    return _bipermutedimsconj(isconj, src, perm_codomain, perm_domain; kwargs...)
end
function _bipermutedimsconj(
        isconj::Bool, src, perm_codomain, perm_domain;
        allocator = DefaultAllocator(), kwargs...
    )
    # `bipermutedimsconjadd!` checks against the destination built from the same permutation, so
    # it cannot catch an invalid one.
    check_biperm(src, perm_codomain, perm_domain)
    dest = allocate_output(
        bipermutedimsconj, isconj, src, perm_codomain, perm_domain; allocator
    )
    return bipermutedimsconjadd!(
        dest, isconj, src, perm_codomain, perm_domain, true, false; kwargs...
    )
end

"""
    permutedimsop(op, src, perm; alg = nothing) -> dest
    permutedimsop(op, src, perm_codomain, perm_domain; alg = nothing) -> dest

Return `op.(permutedims(src, perm))` in new storage, where `op` is `identity` or `conj`. The
permutation is either flat or split into `perm_codomain` and `perm_domain`, as in
[`bipermutedimsop`](@ref).
"""
permutedimsop(op, src, perm; kwargs...) = bipermutedimsop(op, src, perm, (); kwargs...)
function permutedimsop(op, src, perm_codomain, perm_domain; kwargs...)
    return bipermutedimsop(op, src, perm_codomain, perm_domain; kwargs...)
end
"""
    permutedimsconj(isconj::Bool, src, perm; alg = nothing) -> dest
    permutedimsconj(isconj::Bool, src, perm_codomain, perm_domain; alg = nothing) -> dest

Return `permutedims(src, perm)` in new storage, with `src` conjugated when `isconj` is `true`.
The permutation is either flat or split into `perm_codomain` and `perm_domain`, as in
[`bipermutedimsconj`](@ref).
"""
function permutedimsconj(isconj::Bool, src, perm; kwargs...)
    return bipermutedimsconj(isconj, src, perm, (); kwargs...)
end
function permutedimsconj(isconj::Bool, src, perm_codomain, perm_domain; kwargs...)
    return bipermutedimsconj(isconj, src, perm_codomain, perm_domain; kwargs...)
end

"""
    allocate_output(f, args...; allocator = TensorAlgebra.DefaultAllocator())

Allocate the destination that `f` writes its result into when called with `args`, with its
storage from `allocator` through [`similar_map`](@ref). Array types overload it to choose the
type of the output. Algorithms pass their [`TensorAlgebra.allocator`](@ref) for temporaries.
"""
function allocate_output end

function allocate_output(
        ::typeof(bipermutedimsconj), isconj::Bool, src, perm_co, perm_do;
        allocator = DefaultAllocator()
    )
    axes_co = map(i -> isconj ? conj(axes(src, i)) : axes(src, i), perm_co)
    axes_do = map(i -> isconj ? conj(axes(src, i)) : axes(src, i), perm_do)
    # `similar_map` takes the domain axes un-dualized.
    return similar_map(allocator, src, eltype(src), axes_co, conj.(axes_do))
end

"""
    bipermutedims(a, perm_codomain, perm_domain; alg = nothing)
    bipermutedims(a, biperm::BiTuple; alg = nothing)

Return `permutedims(a, (perm_codomain..., perm_domain...))` in new storage, split into the
codomain `perm_codomain` and the domain `perm_domain`.

See also [`bipermutedimsconj`](@ref).
"""
function bipermutedims(a, perm1, perm2; kwargs...)
    return bipermutedimsconj(false, a, perm1, perm2; kwargs...)
end
"""
    bipermutedims!(dest, a, perm_codomain, perm_domain; alg = nothing)
    bipermutedims!(dest, a, biperm::BiTuple; alg = nothing)

In-place counterpart of [`bipermutedims`](@ref): write the permuted `a` into `dest`.
"""
function bipermutedims!(a_dest, a_src, perm1, perm2; kwargs...)
    return bipermutedimsconjadd!(a_dest, false, a_src, perm1, perm2, true, false; kwargs...)
end
function bipermutedims(a, biperm::BiTuple; kwargs...)
    return bipermutedims(a, biperm.t1, biperm.t2; kwargs...)
end
function bipermutedims!(a_dest, a_src, biperm::BiTuple; kwargs...)
    return bipermutedims!(a_dest, a_src, biperm.t1, biperm.t2; kwargs...)
end

# =====================================  matricize  ========================================

"""
    matricizeop(op, a, perm_codomain, perm_domain; alg = nothing) -> m
    matricizeop(op, a, ndims_codomain::Val; alg = nothing) -> m

Matricize `op.(a)`, fusing the codomain dimensions `perm_codomain` into rows and the domain
dimensions `perm_domain` into columns. `op` is `identity` or `conj`. The result may share
`a`'s memory, so treat it as read-only.

See also [`matricizeconj`](@ref) and [`unmatricize`](@ref).
"""
function matricizeop(op, a, perm_codomain, perm_domain; kwargs...)
    return matricizeconj(_isconj(op), a, perm_codomain, perm_domain; kwargs...)
end
"""
    matricizeconj(isconj::Bool, a, perm_codomain, perm_domain; alg = nothing,
                  allocator = TensorAlgebra.DefaultAllocator()) -> m

Matricize `a`, conjugated when `isconj` is `true`, with the dimensions `perm_codomain` as rows
and `perm_domain` as columns. When it copies, `alg` is the [`PermuteAlgorithm`](@ref) and
`allocator` provides the new storage.

See also [`matricizeop`](@ref), [`matricizeconjview`](@ref), [`matricizeconj!`](@ref).
"""
function matricizeconj(isconj::Bool, a, perm_codomain, perm_domain; kwargs...)
    return _matricizeconj(isconj, a, perm_codomain, perm_domain; kwargs...)
end
function _matricizeconj(isconj::Bool, a, perm_codomain, perm_domain; kwargs...)
    check_biperm(a, perm_codomain, perm_domain)
    isoutputview(matricizeconj, isconj, a, perm_codomain, perm_domain) &&
        return matricizeconjview(isconj, a, perm_codomain, perm_domain)
    return matricizeconjcopy(isconj, a, perm_codomain, perm_domain; kwargs...)
end

"""
    matricize(a, perm_codomain, perm_domain; alg = nothing) -> m
    matricize(a, ndims_codomain::Val; alg = nothing) -> m

[`matricizeop`](@ref) with `op = identity`.
"""
function matricize(a, perm_codomain, perm_domain; kwargs...)
    return matricizeconj(false, a, perm_codomain, perm_domain; kwargs...)
end

function matricize(a, ndims_codomain::Val; kwargs...)
    return matricize(a, identitybiperm(ndims_codomain, Val(ndims(a)))...; kwargs...)
end
function matricizeop(op, a, ndims_codomain::Val; kwargs...)
    return matricizeop(op, a, identitybiperm(ndims_codomain, Val(ndims(a)))...; kwargs...)
end

"""
    matricizeopcopy(op, a, perm_codomain, perm_domain; alg = nothing) -> m

Matricize `op.(a)` into new storage that the caller owns. `op` is `identity` or `conj`.

See also [`matricizeconjcopy`](@ref).
"""
function matricizeopcopy(op, a, perm_codomain, perm_domain; kwargs...)
    return matricizeconjcopy(_isconj(op), a, perm_codomain, perm_domain; kwargs...)
end
"""
    matricizeconjcopy(isconj::Bool, a, perm_codomain, perm_domain; alg = nothing,
                      allocator = TensorAlgebra.DefaultAllocator()) -> m

Matricize `a`, conjugated when `isconj` is `true`, into new storage that the caller owns. Array
types customize it through [`allocate_output`](@ref) and [`matricizeconj!`](@ref).

See also [`matricizeopcopy`](@ref).
"""
function matricizeconjcopy(isconj::Bool, a, perm_codomain, perm_domain; kwargs...)
    return _matricizeconjcopy(isconj, a, perm_codomain, perm_domain; kwargs...)
end
function _matricizeconjcopy(
        isconj::Bool, a, perm_codomain, perm_domain;
        allocator = DefaultAllocator(), kwargs...
    )
    check_biperm(a, perm_codomain, perm_domain)
    dest = allocate_output(matricizeconj, isconj, a, perm_codomain, perm_domain; allocator)
    return matricizeconj!(dest, isconj, a, perm_codomain, perm_domain; kwargs...)
end

"""
    matricizeopview(op, a, perm_codomain, perm_domain) -> m

Matricize `op.(a)` as a matrix sharing `a`'s memory. `op` is `identity` or `conj`.

See also [`matricizeconjview`](@ref).
"""
function matricizeopview(op, a, perm_codomain, perm_domain)
    return matricizeconjview(_isconj(op), a, perm_codomain, perm_domain)
end
"""
    matricizeconjview(isconj::Bool, a, perm_codomain, perm_domain) -> m

Matricize `a`, conjugated when `isconj` is `true`, as a matrix sharing `a`'s memory. Array
types overload this for the arguments where [`isoutputview`](@ref) returns `true`.

See also [`matricizeopview`](@ref).
"""
function matricizeconjview(isconj::Bool, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeconjview, (isconj, a, perm_codomain, perm_domain)))
end

"""
    matricizeop!(dest, op, a, perm_codomain, perm_domain; alg = nothing) -> dest

Write the matricization of `op.(a)` into the matrix `dest`. `op` is `identity` or `conj`.

See also [`matricizeconj!`](@ref).
"""
function matricizeop!(dest, op, a, perm_codomain, perm_domain; kwargs...)
    return matricizeconj!(dest, _isconj(op), a, perm_codomain, perm_domain; kwargs...)
end
"""
    matricizeconj!(dest, isconj::Bool, a, perm_codomain, perm_domain; alg = nothing) -> dest

Write the matricization of `a`, conjugated when `isconj` is `true`, into the matrix `dest`.
Array types overload this together with
`allocate_output(matricizeconj, isconj, a, perm_codomain, perm_domain; allocator)`, which
allocates `dest`, and forward the keyword arguments to the permutation.

See also [`matricizeop!`](@ref).
"""
function matricizeconj!(dest, isconj::Bool, a, perm_codomain, perm_domain; kwargs...)
    return throw(MethodError(matricizeconj!, (dest, isconj, a, perm_codomain, perm_domain)))
end

function allocate_output(
        ::typeof(matricizeconj), isconj::Bool, a, perm_codomain, perm_domain; kwargs...
    )
    return throw(
        MethodError(
            allocate_output,
            (matricizeconj, isconj, a, perm_codomain, perm_domain)
        )
    )
end

# ==================================  isoutputview  ======================================
"""
    isoutputview(matricizeconj, isconj::Bool, a, perm_codomain, perm_domain) -> Bool
    isoutputview(unmatricize, m, axes_codomain, axes_domain) -> Bool

Whether `matricizeconj` (or `unmatricize`) returns an array sharing its input's memory for these
arguments. Defaults to `false`. Array types overload it together with
[`matricizeconjview`](@ref) (or [`unmatricizeview`](@ref)).
"""
function isoutputview(
        ::typeof(matricizeconj), isconj::Bool, a, perm_codomain, perm_domain
    )
    return false
end

# =====================================  unmatricize  ======================================

"""
    unmatricize(m, axes_codomain, axes_domain; alg = nothing) -> a

Unfold the matrix `m` into an array with axes `(axes_codomain..., conj.(axes_domain)...)`,
undoing [`matricize`](@ref) with the dimensions in order. The result may share `m`'s memory, so
treat it as read-only.

See also [`unmatricizeview`](@ref), [`unmatricizecopy`](@ref), [`unmatricizeadd!`](@ref).
"""
function unmatricize(m, axes_codomain, axes_domain; kwargs...)
    isoutputview(unmatricize, m, axes_codomain, axes_domain) &&
        return unmatricizeview(m, axes_codomain, axes_domain)
    return unmatricizecopy(m, axes_codomain, axes_domain; kwargs...)
end

"""
    unmatricizecopy(m, axes_codomain, axes_domain; alg = nothing,
                    allocator = TensorAlgebra.DefaultAllocator()) -> a

Unfold the matrix `m` like [`unmatricize`](@ref), into new storage from `allocator`. Array
types customize it through `allocate_output(unmatricize, m, axes_codomain, axes_domain; allocator)` and [`unmatricizeadd!`](@ref).
"""
function unmatricizecopy(
        m, axes_codomain, axes_domain; allocator = DefaultAllocator(), kwargs...
    )
    a = allocate_output(unmatricize, m, axes_codomain, axes_domain; allocator)
    return unmatricize!(a, m, Val(length(axes_codomain)); kwargs...)
end

"""
    unmatricizeview(m, axes_codomain, axes_domain) -> a

Unfold the matrix `m` like [`unmatricize`](@ref), as an array sharing `m`'s memory. Array types
overload this for the arguments where [`isoutputview`](@ref) returns `true`, and
[`unmatricizeadd!`](@ref) reads the matrix through it.
"""
function unmatricizeview(m, axes_codomain, axes_domain)
    return throw(MethodError(unmatricizeview, (m, axes_codomain, axes_domain)))
end

function isoutputview(::typeof(unmatricize), m, axes_codomain, axes_domain)
    return false
end

"""
    bipartition_axes(t::Tuple, length1::Val) -> (axes_codomain, axes_domain)
    bipartition_axes(t::Tuple, group1::Tuple, group2::Tuple) -> (axes_codomain, axes_domain)

Split the axes `t` like [`bipartition`](@ref), with the domain axes conjugated into the form
[`unmatricize`](@ref) and [`similar_map`](@ref) take.
"""
function bipartition_axes(t::Tuple, split...)
    axes_codomain, axes_domain = bipartition(t, split...)
    return axes_codomain, conj.(axes_domain)
end

"""
    unmatricizeadd!(a_dest, m, perm_codomain, perm_domain, α, β; alg = nothing) -> a_dest

`a_dest = β * a_dest + α * a`, where `a` is the array whose matricization with
`(perm_codomain, perm_domain)` is `m`. Array types overload this together with
`allocate_output(unmatricize, m, axes_codomain, axes_domain; allocator)`, which allocates
`a_dest`, and forward the keyword arguments to the permutation.
"""
function unmatricizeadd!(
        a_dest, m,
        perm_codomain, perm_domain,
        α::Number, β::Number;
        kwargs...
    )
    biperm_src = BiTuple(perm_codomain, perm_domain)
    ndims(a_dest) == length(biperm_src) ||
        throw(ArgumentError("destination does not match permutation"))
    axes_codomain, axes_domain = bipartition_axes(axes(a_dest), biperm_src)
    a_perm = unmatricizeview(m, axes_codomain, axes_domain)
    biperm_dest = invperm(biperm_src)
    return bipermutedimsconjadd!(
        a_dest,
        false,
        a_perm,
        biperm_dest.t1,
        biperm_dest.t2,
        α,
        β;
        kwargs...
    )
end

"""
    unmatricize!(a_dest, m, perm_codomain, perm_domain; alg = nothing) -> a_dest
    unmatricize!(a_dest, m, ndims_codomain::Val; alg = nothing) -> a_dest

Overwrite `a_dest` with the array whose matricization with the same arguments is `m`.

See also [`unmatricizeadd!`](@ref).
"""
function unmatricize!(a_dest, m, perm_codomain, perm_domain; kwargs...)
    return unmatricizeadd!(a_dest, m, perm_codomain, perm_domain, true, false; kwargs...)
end

function unmatricize!(a_dest, m, ndims_codomain::Val; kwargs...)
    return unmatricize!(
        a_dest, m, identitybiperm(ndims_codomain, Val(ndims(a_dest)))...; kwargs...
    )
end

# ================================  dense (reshape) hooks  ===================================
# Only a `DenseArray` declares a view, since reshaping any other array gives a wrapper that `mul!`
# does not send to BLAS, and that is slow to read. The view is the reshape when the dimensions are
# already in order, and its `transpose`, or `adjoint` when conjugating, at the swapped split.
function isoutputview(
        ::typeof(matricizeconj), isconj::Bool, a::DenseArray, perm_codomain, perm_domain
    )
    return !isconj && isidentitybiperm(perm_codomain, perm_domain) ||
        isidentitybiperm(perm_domain, perm_codomain)
end
function matricizeconjview(isconj::Bool, a::DenseArray, perm_codomain, perm_domain)
    length_codomain = prod(i -> size(a, i), perm_codomain; init = 1)
    length_domain = prod(i -> size(a, i), perm_domain; init = 1)
    !isconj && isidentitybiperm(perm_codomain, perm_domain) &&
        return reshape(a, (length_codomain, length_domain))
    m = reshape(a, (length_domain, length_codomain))
    return isconj ? adjoint(m) : transpose(m)
end
function allocate_output(
        ::typeof(matricizeconj), isconj::Bool, a::AbstractArray, perm_codomain, perm_domain;
        allocator = DefaultAllocator()
    )
    length_codomain = prod(i -> size(a, i), perm_codomain; init = 1)
    length_domain = prod(i -> size(a, i), perm_domain; init = 1)
    return similar_map(
        allocator, a, eltype(a), (Base.OneTo(length_codomain),),
        (Base.OneTo(length_domain),)
    )
end
function matricizeconj!(
        dest, isconj::Bool, a::AbstractArray, perm_codomain, perm_domain; kwargs...
    )
    perm = (perm_codomain..., perm_domain...)
    dest_tensor = reshape(dest, map(i -> size(a, i), perm))
    bipermutedimsconjadd!(
        dest_tensor, isconj, a, perm_codomain, perm_domain, true, false; kwargs...
    )
    return dest
end
# `reshape` alone only checks the total length, so it would accept a wrong split.
function check_input(::typeof(unmatricize), m, axes_codomain, axes_domain)
    (
        ndims(m) == 2 &&
            size(m, 1) == prod(length, axes_codomain; init = 1) &&
            size(m, 2) == prod(length, axes_domain; init = 1)
    ) || throw(DimensionMismatch("`unmatricize` axes do not match the matrix size"))
    return nothing
end
isoutputview(::typeof(unmatricize), m::DenseArray, axes_codomain, axes_domain) = true
function unmatricizeview(m::AbstractMatrix, axes_codomain, axes_domain)
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return reshape(m, (axes_codomain..., conj.(axes_domain)...))
end
function allocate_output(
        ::typeof(unmatricize),
        m::AbstractMatrix,
        axes_codomain,
        axes_domain;
        allocator = DefaultAllocator()
    )
    return similar_map(allocator, m, eltype(m), axes_codomain, axes_domain)
end
