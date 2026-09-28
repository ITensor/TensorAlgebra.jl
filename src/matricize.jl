using LinearAlgebra: Diagonal

# =======================================  misc  ========================================

"""
    TensorAlgebra.trivialrange(R::Type{<:AbstractUnitRange}[, n::Integer])
    TensorAlgebra.trivialrange(r::AbstractUnitRange[, n::Integer])

Return the identity range for fusing ranges of type `R`: a one-dimensional range
`t` for which fusing `t` with any other range of the same family leaves that range
unchanged. Defaults to `Base.OneTo(1)`. Downstream packages overload the type-level
methods to return their own identity (for example, a charge-0 one-dimensional sector
for a graded range).

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
"""
function permutedimsop(op, src, perm_codomain, perm_domain)
    # Validate against `src` here: `bipermutedimsopadd!`'s `check_input` compares against `dest`,
    # which `allocate_output` builds from the same perms, so it cannot catch a non-covering perm.
    check_biperm(src, perm_codomain, perm_domain)
    dest = allocate_output(permutedimsop, op, src, perm_codomain, perm_domain)
    return bipermutedimsopadd!(dest, op, src, perm_codomain, perm_domain, true, false)
end

# The output holds `op.(src)` permuted, so `op` applies to the axes too: `conj` dualizes a
# graded axis (a no-op on a dense axis), `identity` leaves it unchanged, keeping axes and
# data in sync.
function allocate_output(::typeof(permutedimsop), op, src, perm_co, perm_do)
    T = Base.promote_op(op, eltype(src))
    axes_co = map(i -> op(axes(src, i)), perm_co)
    axes_do = map(i -> op(axes(src, i)), perm_do)
    # `axes_do` are in the stored/dualized convention (from `axes(src)`), so un-dualize them
    # into `similar_map`'s codomain-facing convention.
    return similar_map(src, T, axes_co, conj.(axes_do))
end

function bipermutedims(a, perm1, perm2)
    return permutedimsop(identity, a, perm1, perm2)
end
function bipermutedims!(a_dest, a_src, perm1, perm2)
    return bipermutedimsopadd!(a_dest, identity, a_src, perm1, perm2, true, false)
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
# An array type implements four hooks, all taking the operation, the array and the bipermutation:
#
#   `allocate_output(matricizeop, op, a, pc, pd)`  the matrix destination
#   `matricizeop!(dest, op, a, pc, pd)`            write the matricization into it
#   `matricizeopview(op, a, pc, pd)`               partial: the aliasing form
#   `is_output_view(matricizeop, op, a, pc, pd)`   whether the aliasing form applies
#
# Everything else is derived. `matricizeopcopy` allocates and writes, so it always returns fresh
# storage the caller owns. `matricizeop` returns the view where the array type declares one and
# the copy otherwise, i.e. it has maybe-alias semantics and its result must be treated as
# read-only. `matricize` is `matricizeop` at `identity`.
#
# Allocation is a hook rather than generic machinery because computing a matricized destination
# needs the fused axes, which only the array type knows: TensorAlgebra deliberately has no generic
# axis-fusion interface. It is also what makes the copy path terminate, since `matricizeop!` is a
# distinct function from the router rather than a re-entry into it.
#
# `matricizeopcopy` is itself an overload point for an array type whose owned matricization
# already falls out of an allocating operation it has (for a graded array, permuting into fresh
# storage whose stored matrix is the answer). Such a type overloads the copy instead of
# `allocate_output`/`matricizeop!`, which would copy that storage a second time, and then owes
# only `matricizeopview` and `is_output_view`.

"""
    matricizeop(op, a, perm_codomain, perm_domain)

Matricize `a` across the bipermutation with the element-wise operation `op` folded in, i.e. a
matrix representing `op.(permutedims(a, (perm_codomain..., perm_domain...)))` with the codomain
fused to rows and the domain to columns.

Has "maybe alias" semantics: the result may share `a`'s memory or be fresh storage, depending on
the array type. Treat it as read-only. Use `matricizeopcopy` for a matrix the caller owns, and
`matricizeopview` (partial) for one guaranteed to alias.
"""
function matricizeop(op, a, perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    is_output_view(matricizeop, op, a, perm_codomain, perm_domain) &&
        return matricizeopview(op, a, perm_codomain, perm_domain)
    return matricizeopcopy(op, a, perm_codomain, perm_domain)
end

"""
    matricize(a, perm_codomain, perm_domain)

`matricizeop` at `identity`. Has the same maybe-alias semantics.
"""
function matricize(a, perm_codomain, perm_domain)
    return matricizeop(identity, a, perm_codomain, perm_domain)
end

# Split-only convenience: matricize after `ndims_codomain` dimensions without permuting. Sugar over
# the bipermutation forms, not a dispatch tier. An array type implements the hooks above and never
# these, which is what keeps the copy path from recursing back through the router.
function matricize(a, ndims_codomain::Val)
    return matricize(a, identitybiperm(ndims_codomain, Val(ndims(a)))...)
end
function matricizeop(op, a, ndims_codomain::Val)
    return matricizeop(op, a, identitybiperm(ndims_codomain, Val(ndims(a)))...)
end

# Total: always fresh storage the caller owns.
function matricizeopcopy(op, a, perm_codomain, perm_domain)
    check_biperm(a, perm_codomain, perm_domain)
    dest = allocate_output(matricizeop, op, a, perm_codomain, perm_domain)
    return matricizeop!(dest, op, a, perm_codomain, perm_domain)
end

# Partial: defined only where `is_output_view` is `true`, and always returns a matricization
# sharing `a`'s memory (the `StridedView` partial-constructor pattern).
function matricizeopview(op, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeopview, (op, a, perm_codomain, perm_domain)))
end

# Required of every array type: write the matricization of `a` into `dest`.
function matricizeop!(dest, op, a, perm_codomain, perm_domain)
    return throw(MethodError(matricizeop!, (dest, op, a, perm_codomain, perm_domain)))
end

# Required of every array type: the matrix destination `matricizeop!` writes into.
function allocate_output(::typeof(matricizeop), op, a, perm_codomain, perm_domain)
    return throw(
        MethodError(allocate_output, (matricizeop, op, a, perm_codomain, perm_domain))
    )
end

# ==================================  is_output_view  ======================================
# `true` iff `matricizeop(op, a, perm_codomain, perm_domain)` shares `a`'s memory, so that
# writes to the result are writes to `a`. The `isstrided`/`StridedView` pattern, and the same
# question TensorKit asks with `has_shared_permute` and TensorOperations with `isblasdestination`.
# Keyed on the operation like the other function-keyed hooks (`check_input`, `allocate_output`,
# `output_axes`), so the predicate's arguments are exactly the call's arguments. The untyped
# default declares nothing, which is always safe.
function is_output_view(::typeof(matricizeop), op, a, perm_codomain, perm_domain)
    return false
end

# Split form: `axes_codomain` and `axes_domain` are the destination axes for the codomain and
# domain groups, given codomain-facing (un-dualized), the same convention as `similar_map`. A
# matrix type that stores its domain axes dualized re-dualizes them with `conj` in its overload
# (a no-op on a dense axis). This is the primary overload point for a new matrix type, dispatched
# on `m`. Permutation is handled by the bipermutation form of `unmatricize!`, so out-of-place
# `unmatricize` never has to disambiguate axis tuples from permutation tuples regardless of how
# unconstrained `m` and the axes are.
function unmatricize end

# Split `axes` into its codomain and domain groups like `bipartition`, but present the domain
# group codomain-facing (un-dualized) with `conj`, the convention `unmatricize` and `similar_map`
# take. The domain axes `bipartition` reads off `axes(a)` are in the stored (dualized) form, so
# this bridges from `axes(a)` to the `unmatricize` axis convention (a no-op on dense axes).
function bipartition_axes(t::Tuple, split...)
    axes_codomain, axes_domain = bipartition(t, split...)
    return axes_codomain, conj.(axes_domain)
end

# `a_dest = β * a_dest + α * unmatricize(m)`, scattered across the bipermutation in one pass.
# The bipermutation maps the destination's dimension order to the matrix's: `axes(a_dest)` grouped
# by it gives the legs in `m`'s order, and the result is permuted back by its inverse. It is not
# intrinsically an inverse permutation — the matricized-contraction destination path happens to
# derive it as `invperm(biperm_dest)`, while a `matricize`/`unmatricize!` round trip passes
# the same forward bipermutation to both. Dispatched on `a_dest`: a wrapper type scatters into its
# parent by overloading this form, and `unmatricize!` is the `(1, 0)` case.
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
    return bipermutedimsopadd!(
        a_dest,
        identity,
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

# In-place counterpart of `unmatricize`:
# scatter the fused matrix `m` back into `a_dest`'s existing storage across the codomain/domain
# split at `ndims_codomain`. The split applies no permutation, so this is the bipermutation form at the
# trivial bipermutation, reusing its in-place block scatter (no intermediate `unmatricize` copy).
function unmatricize!(a_dest, m, ndims_codomain::Val)
    return unmatricize!(a_dest, m, identitybiperm(ndims_codomain, Val(ndims(a_dest)))...)
end

# ================================  dense (reshape) hooks  ===================================
# The view must be a matrix that BLAS and LAPACK handle efficiently, not merely one that shares
# memory. A `DenseArray` (`Array`, GPU arrays) reshapes to one of its own kind, so it declares the
# view; any other `AbstractArray` reshapes to a `ReshapedArray` wrapper that `mul!` sends to
# generic matmul, so it takes the owned copy instead and lands on BLAS. The reshape shares memory
# only when the data is already in codomain-then-domain order and no operation has to be folded
# in: a reshape can neither reorder nor carry a `conj`. The copy hooks below stay generic, since
# `similar` plus a permuted add is a correct owned matricization for any permutable array.
function is_output_view(
        ::typeof(matricizeop), op, a::DenseArray, perm_codomain, perm_domain
    )
    return op === identity && isidentitybiperm(perm_codomain, perm_domain)
end
function matricizeopview(op, a::DenseArray, perm_codomain, perm_domain)
    size_codomain, size_domain = bipartition(size(a), Val(length(perm_codomain)))
    return reshape(a, (prod(size_codomain), prod(size_domain)))
end
function allocate_output(
        ::typeof(matricizeop), op, a::AbstractArray, perm_codomain, perm_domain
    )
    T = Base.promote_op(op, eltype(a))
    size_codomain = map(i -> size(a, i), perm_codomain)
    size_domain = map(i -> size(a, i), perm_domain)
    return similar(a, T, (prod(size_codomain), prod(size_domain)))
end
# The destination is a dense matrix, so reshaping it to the permuted tensor shape is a view and
# the permuted-add writes straight through it.
function matricizeop!(dest, op, a::AbstractArray, perm_codomain, perm_domain)
    perm = (perm_codomain..., perm_domain...)
    dest_tensor = reshape(dest, map(i -> size(a, i), perm))
    bipermutedimsopadd!(dest_tensor, op, a, perm_codomain, perm_domain, true, false)
    return dest
end
# The matricized input's rows must be the fused codomain and its columns the fused domain.
# `reshape` alone only checks the total element count, so a wrong split with the right total
# would reshape silently.
function check_input(::typeof(unmatricize), m, axes_codomain, axes_domain)
    (
        ndims(m) == 2 &&
            size(m, 1) == prod(length, axes_codomain; init = 1) &&
            size(m, 2) == prod(length, axes_domain; init = 1)
    ) || throw(DimensionMismatch("`unmatricize` axes do not match the matrix size"))
    return nothing
end
# A dense reshape ignores the codomain/domain split: it just reshapes to the concatenated axes.
# `conj` re-dualizes the codomain-facing `axes_domain` into stored form, a no-op on a dense axis.
function unmatricize(m::AbstractMatrix, axes_codomain, axes_domain)
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return reshape(m, (axes_codomain..., conj.(axes_domain)...))
end
