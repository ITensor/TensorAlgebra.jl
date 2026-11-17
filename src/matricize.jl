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
    permutedimsop(op, src, perm_codomain, perm_domain) -> dest

Return `op.(permutedims(src, (perm_codomain..., perm_domain...)))` in new storage, split into
the codomain `perm_codomain` and the domain `perm_domain`. `op` is `identity` or `conj`.

See also [`permutedimsconj`](@ref).
"""
function permutedimsop(op, src, perm_codomain, perm_domain)
    return permutedimsconj(_isconj(op), src, perm_codomain, perm_domain)
end
"""
    permutedimsconj(isconj::Bool, src, perm_codomain, perm_domain) -> dest

Return `permutedims(src, (perm_codomain..., perm_domain...))` in new storage, split into the
codomain `perm_codomain` and the domain `perm_domain`, with `src` conjugated when `isconj` is
`true`. Array types overload this function to customize it.

See also [`permutedimsop`](@ref).
"""
function permutedimsconj(isconj::Bool, src, perm_codomain, perm_domain)
    src isa ConjArray &&
        return permutedimsconj(!isconj, parent(src), perm_codomain, perm_domain)
    # `bipermutedimsconjadd!` checks against the destination built from the same permutation, so
    # it cannot catch an invalid one.
    check_biperm(src, perm_codomain, perm_domain)
    dest = allocate_output(permutedimsconj, isconj, src, perm_codomain, perm_domain)
    return bipermutedimsconjadd!(dest, isconj, src, perm_codomain, perm_domain, true, false)
end

"""
    allocate_output(f, args...)

Allocate the destination that `f` writes its result into when called with `args`. Array types
overload it to choose the type of the output.
"""
function allocate_output end

function allocate_output(::typeof(permutedimsconj), isconj::Bool, src, perm_co, perm_do)
    axes_co = map(i -> isconj ? conj(axes(src, i)) : axes(src, i), perm_co)
    axes_do = map(i -> isconj ? conj(axes(src, i)) : axes(src, i), perm_do)
    # `similar_map` takes the domain axes un-dualized.
    return similar_map(src, eltype(src), axes_co, conj.(axes_do))
end

function bipermutedims(a, perm1, perm2)
    return permutedimsconj(false, a, perm1, perm2)
end
function bipermutedims!(a_dest, a_src, perm1, perm2)
    return bipermutedimsconjadd!(a_dest, false, a_src, perm1, perm2, true, false)
end
function bipermutedims(a, biperm::BiTuple)
    return bipermutedims(a, biperm.t1, biperm.t2)
end
function bipermutedims!(
        a_dest, a_src, biperm::BiTuple
    )
    return bipermutedims!(a_dest, a_src, biperm.t1, biperm.t2)
end

# =====================================  matricize  ========================================

"""
    matricizeop(op, a, perm_codomain, perm_domain) -> m
    matricizeop(op, a, ndims_codomain::Val) -> m

Matricize `op.(a)`, fusing the codomain dimensions `perm_codomain` into rows and the domain
dimensions `perm_domain` into columns. `op` is `identity` or `conj`. The result may share
`a`'s memory, so treat it as read-only.

See also [`matricizeconj`](@ref) and [`unmatricize`](@ref).
"""
function matricizeop(op, a, perm_codomain, perm_domain)
    return matricizeconj(_isconj(op), a, perm_codomain, perm_domain)
end
"""
    matricizeconj(isconj::Bool, a, perm_codomain, perm_domain) -> m

Matricize `a`, conjugated when `isconj` is `true`, with the dimensions `perm_codomain` as rows
and `perm_domain` as columns. Array types overload this function to customize it.

See also [`matricizeop`](@ref), [`matricizeconjview`](@ref), [`matricizeconj!`](@ref).
"""
function matricizeconj(isconj::Bool, a, perm_codomain, perm_domain)
    a isa ConjArray && return matricizeconj(!isconj, parent(a), perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    is_output_view(matricizeconj, isconj, a, perm_codomain, perm_domain) &&
        return matricizeconjview(isconj, a, perm_codomain, perm_domain)
    return matricizeconjcopy(isconj, a, perm_codomain, perm_domain)
end

"""
    matricize(a, perm_codomain, perm_domain) -> m
    matricize(a, ndims_codomain::Val) -> m

[`matricizeop`](@ref) with `op = identity`.
"""
function matricize(a, perm_codomain, perm_domain)
    return matricizeconj(false, a, perm_codomain, perm_domain)
end

function matricize(a, ndims_codomain::Val)
    return matricize(a, identitybiperm(ndims_codomain, Val(ndims(a)))...)
end
function matricizeop(op, a, ndims_codomain::Val)
    return matricizeop(op, a, identitybiperm(ndims_codomain, Val(ndims(a)))...)
end

"""
    matricizeopcopy(op, a, perm_codomain, perm_domain) -> m

Matricize `op.(a)` into new storage that the caller owns. `op` is `identity` or `conj`.

See also [`matricizeconjcopy`](@ref).
"""
function matricizeopcopy(op, a, perm_codomain, perm_domain)
    return matricizeconjcopy(_isconj(op), a, perm_codomain, perm_domain)
end
"""
    matricizeconjcopy(isconj::Bool, a, perm_codomain, perm_domain) -> m

Matricize `a`, conjugated when `isconj` is `true`, into new storage that the caller owns.

See also [`matricizeopcopy`](@ref).
"""
function matricizeconjcopy(isconj::Bool, a, perm_codomain, perm_domain)
    a isa ConjArray &&
        return matricizeconjcopy(!isconj, parent(a), perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    dest = allocate_output(matricizeconj, isconj, a, perm_codomain, perm_domain)
    return matricizeconj!(dest, isconj, a, perm_codomain, perm_domain)
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
types overload this for the arguments where [`is_output_view`](@ref) returns `true`.

See also [`matricizeopview`](@ref).
"""
function matricizeconjview(isconj::Bool, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeconjview, (isconj, a, perm_codomain, perm_domain)))
end

"""
    matricizeop!(dest, op, a, perm_codomain, perm_domain) -> dest

Write the matricization of `op.(a)` into the matrix `dest`. `op` is `identity` or `conj`.

See also [`matricizeconj!`](@ref).
"""
function matricizeop!(dest, op, a, perm_codomain, perm_domain)
    return matricizeconj!(dest, _isconj(op), a, perm_codomain, perm_domain)
end
"""
    matricizeconj!(dest, isconj::Bool, a, perm_codomain, perm_domain) -> dest

Write the matricization of `a`, conjugated when `isconj` is `true`, into the matrix `dest`.
Array types overload this together with [`allocate_output`](@ref), which allocates `dest`.

See also [`matricizeop!`](@ref).
"""
function matricizeconj!(dest, isconj::Bool, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeconj!, (dest, isconj, a, perm_codomain, perm_domain)))
end

function allocate_output(
        ::typeof(matricizeconj), isconj::Bool, a, perm_codomain, perm_domain
    )
    return throw(
        MethodError(
            allocate_output,
            (matricizeconj, isconj, a, perm_codomain, perm_domain)
        )
    )
end

# ==================================  is_output_view  ======================================
"""
    is_output_view(matricizeconj, isconj::Bool, a, perm_codomain, perm_domain) -> Bool

Whether `matricizeconj` returns a matrix sharing `a`'s memory for these arguments. Defaults to
`false`. Array types overload it together with [`matricizeconjview`](@ref).
"""
function is_output_view(
        ::typeof(matricizeconj), isconj::Bool, a, perm_codomain, perm_domain
    )
    return false
end

"""
    unmatricize(m, axes_codomain, axes_domain) -> a

Unfold the matrix `m` into an array with axes `(axes_codomain..., conj.(axes_domain)...)`,
undoing [`matricize`](@ref) with the dimensions in order. Array types overload this.
"""
function unmatricize end

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
    unmatricizeadd!(a_dest, m, perm_codomain, perm_domain, α, β) -> a_dest

`a_dest = β * a_dest + α * a`, where `a` is the array whose matricization with
`(perm_codomain, perm_domain)` is `m`.
"""
function unmatricizeadd!(
        a_dest, m,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    biperm_src = BiTuple(perm_codomain, perm_domain)
    ndims(a_dest) == length(biperm_src) ||
        throw(ArgumentError("destination does not match permutation"))
    axes_codomain, axes_domain = bipartition_axes(axes(a_dest), biperm_src)
    a_perm = unmatricize(m, axes_codomain, axes_domain)
    biperm_dest = BiTuple(Tuple(invperm(biperm_src)), Val(length_codomain(biperm_src)))
    return bipermutedimsconjadd!(
        a_dest,
        false,
        a_perm,
        biperm_dest.t1,
        biperm_dest.t2,
        α,
        β
    )
end

"""
    unmatricize!(a_dest, m, perm_codomain, perm_domain) -> a_dest
    unmatricize!(a_dest, m, ndims_codomain::Val) -> a_dest

Overwrite `a_dest` with the array whose matricization with the same arguments is `m`.

See also [`unmatricizeadd!`](@ref).
"""
function unmatricize!(a_dest, m, perm_codomain, perm_domain)
    return unmatricizeadd!(a_dest, m, perm_codomain, perm_domain, true, false)
end

function unmatricize!(a_dest, m, ndims_codomain::Val)
    return unmatricize!(a_dest, m, identitybiperm(ndims_codomain, Val(ndims(a_dest)))...)
end

# ================================  dense (reshape) hooks  ===================================
# Only a `DenseArray` declares a view, since reshaping any other array gives a wrapper that `mul!`
# does not send to BLAS. The view is the reshape when the dimensions are already in order, and its
# `transpose`, or `adjoint` when conjugating, at the swapped split.
function is_output_view(
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
        ::typeof(matricizeconj), isconj::Bool, a::AbstractArray, perm_codomain, perm_domain
    )
    size_codomain = map(i -> size(a, i), perm_codomain)
    size_domain = map(i -> size(a, i), perm_domain)
    return similar(a, eltype(a), (prod(size_codomain), prod(size_domain)))
end
function matricizeconj!(
        dest, isconj::Bool, a::AbstractArray, perm_codomain, perm_domain
    )
    perm = (perm_codomain..., perm_domain...)
    dest_tensor = reshape(dest, map(i -> size(a, i), perm))
    bipermutedimsconjadd!(dest_tensor, isconj, a, perm_codomain, perm_domain, true, false)
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
function unmatricize(m::AbstractMatrix, axes_codomain, axes_domain)
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return reshape(m, (axes_codomain..., conj.(axes_domain)...))
end
