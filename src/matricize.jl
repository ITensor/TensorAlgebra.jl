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
    permutedimsop(op, src, perm_codomain, perm_domain)

Non-mutating version of `bipermutedimsopadd!`: returns
`op.(permutedims(src, (perm_codomain..., perm_domain...)))`.

See also [`permutedimsconj`](@ref).
"""
function permutedimsop(op, src, perm_codomain, perm_domain)
    return permutedimsconj(_isconj(op), src, perm_codomain, perm_domain)
end
"""
    permutedimsconj(isconj::Bool, src, perm_codomain, perm_domain)

Like [`permutedimsop`](@ref), with the operation given as a conjugation flag.
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
# An array type implements four hooks, all taking the conjugation flag, the array and the
# bipermutation:
#
#   `allocate_output(matricizeconj, isconj, a, pc, pd)`  the matrix destination
#   `matricizeconj!(dest, isconj, a, pc, pd)`            write the matricization into it
#   `matricizeconjview(isconj, a, pc, pd)`               partial: the aliasing form
#   `is_output_view(matricizeconj, isconj, a, pc, pd)`   whether the aliasing form applies
#
# `matricizeconjcopy` and `matricizeconj` are derived from these. A type that already has an
# allocating matricization can overload `matricizeconjcopy` instead of the first two.

"""
    matricizeop(op, a, perm_codomain, perm_domain)

Matricize `a` across the bipermutation with the element-wise operation `op` folded in, i.e. a
matrix representing `op.(permutedims(a, (perm_codomain..., perm_domain...)))` with the codomain
fused to rows and the domain to columns.

The result may share `a`'s memory or be fresh storage, depending on the array type, so treat it
as read-only.

See also [`matricizeconj`](@ref).
"""
function matricizeop(op, a, perm_codomain, perm_domain)
    return matricizeconj(_isconj(op), a, perm_codomain, perm_domain)
end
"""
    matricizeconj(isconj::Bool, a, perm_codomain, perm_domain)

Like [`matricizeop`](@ref), with the operation given as a conjugation flag.
"""
function matricizeconj(isconj::Bool, a, perm_codomain, perm_domain)
    a isa ConjArray && return matricizeconj(!isconj, parent(a), perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    is_output_view(matricizeconj, isconj, a, perm_codomain, perm_domain) &&
        return matricizeconjview(isconj, a, perm_codomain, perm_domain)
    return matricizeconjcopy(isconj, a, perm_codomain, perm_domain)
end

"""
    matricize(a, perm_codomain, perm_domain)

`matricizeop` at `identity`.
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

function matricizeopcopy(op, a, perm_codomain, perm_domain)
    return matricizeconjcopy(_isconj(op), a, perm_codomain, perm_domain)
end
# Always returns fresh storage the caller owns.
function matricizeconjcopy(isconj::Bool, a, perm_codomain, perm_domain)
    a isa ConjArray &&
        return matricizeconjcopy(!isconj, parent(a), perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    dest = allocate_output(matricizeconj, isconj, a, perm_codomain, perm_domain)
    return matricizeconj!(dest, isconj, a, perm_codomain, perm_domain)
end

function matricizeopview(op, a, perm_codomain, perm_domain)
    return matricizeconjview(_isconj(op), a, perm_codomain, perm_domain)
end
# Defined only where `is_output_view` is `true`, and returns a matricization sharing `a`'s memory.
function matricizeconjview(isconj::Bool, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeconjview, (isconj, a, perm_codomain, perm_domain)))
end

function matricizeop!(dest, op, a, perm_codomain, perm_domain)
    return matricizeconj!(dest, _isconj(op), a, perm_codomain, perm_domain)
end
# Required of every array type: write the matricization of `a` into `dest`.
function matricizeconj!(dest, isconj::Bool, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeconj!, (dest, isconj, a, perm_codomain, perm_domain)))
end

# Required of every array type: the matrix destination `matricizeconj!` writes into.
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
# Whether `matricizeconj(isconj, a, perm_codomain, perm_domain)` shares `a`'s memory.
function is_output_view(
        ::typeof(matricizeconj), isconj::Bool, a, perm_codomain, perm_domain
    )
    return false
end

# `axes_domain` is given un-dualized, the same convention as `similar_map`. Permutation is left to
# `unmatricize!`, so `unmatricize` never has to tell axis tuples from permutation tuples.
function unmatricize end

# Like `bipartition`, but returns the domain axes un-dualized, the convention `unmatricize` and
# `similar_map` take.
function bipartition_axes(t::Tuple, split...)
    axes_codomain, axes_domain = bipartition(t, split...)
    return axes_codomain, conj.(axes_domain)
end

# `a_dest = β * a_dest + α * unmatricize(m)`, where the bipermutation maps `a_dest`'s dimensions
# to `m`'s rows and columns, the same one passed to `matricize`.
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
