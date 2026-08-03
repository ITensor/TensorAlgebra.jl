# Throw unless `sz1` and `sz2` are equal ignoring trailing length-1 axes: an axis beyond one
# size's rank counts as length 1, mirroring `Base.size(A, d)` for `d > ndims(A)`. Guards a
# `reshape` against silently reinterpreting same-length data of a genuinely different shape, while
# still allowing trailing length-1 axes to be added or dropped (the projection verbs' "omit
# trailing length-1 axes" convention).
function check_project_size(sz1::Dims, sz2::Dims)
    all(i -> get(sz1, i, 1) == get(sz2, i, 1), 1:max(length(sz1), length(sz2))) || throw(
        DimensionMismatch("sizes $sz1 and $sz2 differ beyond trailing length-1 axes")
    )
    return nothing
end

"""
    projectto!(dest, src) -> dest

Project `src` into the restricted space of `dest` without checking which
components may have been projected out. The default reshapes `src` to
`size(dest)` up to trailing length-1 axes (so a lower-rank `src` may omit
them, e.g. an auxiliary flux-canceling leg a codomain/domain split introduces
on a symmetric state) and `copyto!`s, throwing on a genuine shape mismatch
rather than reinterpreting the data. A backend whose arrays are not
`copyto!`-compatible with a dense array overloads this. This is the in-place
fill primitive that [`unchecked_project`](@ref) allocates a destination for.
"""
function projectto!(dest, src)
    check_project_size(size(src), size(dest))
    return copyto!(dest, reshape(src, size(dest)))
end

"""
    allocate_project(raw, codomain_axes, domain_axes) -> dest

Allocate the destination that projecting `raw` onto
`codomain_axes`/`domain_axes` fills. This is a backend customization point
(with [`projectto!`](@ref) and [`is_projected`](@ref)); the default is plain
`similar_map(raw, codomain_axes, domain_axes)`.

`project` projects into exactly the given axes, so `raw` must not have more
axes than they account for. To append a derived flux-carrying auxiliary axis
for a charge-shifting operator or a non-invariant state, use
[`project_aux`](@ref) instead.
"""
function allocate_project(raw, codomain_axes, domain_axes)
    nphys = length(codomain_axes) + length(domain_axes)
    ndims(raw) <= nphys || throw(
        ArgumentError(
            "`project` projects into exactly the given axes and does not derive an auxiliary \
            axis; got a rank-$(ndims(raw)) input for $nphys given axes. Use `project_aux` to \
            append a derived flux-carrying leg, or pass the axis explicitly."
        )
    )
    return similar_map(raw, codomain_axes, domain_axes)
end

"""
    unchecked_project(raw, codomain_axes, domain_axes) -> dest
    unchecked_project(raw, axes) -> dest

Project `raw` into a symmetry-restricted array shaped as a map from
`domain_axes` to `codomain_axes`, without checking which components are
discarded: entries of `raw` outside the symmetry-allowed structure are
dropped without inspection. Most callers want [`project`](@ref), which
verifies that nothing was discarded, or [`tryproject`](@ref), its nullable
sibling. All three derive from the backend customization points: this one is
`projectto!(allocate_project(raw, codomain_axes, domain_axes), raw)`. The
two-argument form takes a flat list of `axes` and is equivalent to an empty
domain.
"""
function unchecked_project(raw, codomain_axes, domain_axes)
    return projectto!(allocate_project(raw, codomain_axes, domain_axes), raw)
end
# The flat all-codomain (state) form: a list of `axes` with an empty domain.
unchecked_project(raw, axes) = unchecked_project(raw, axes, ())

# The codomain rank a destination reports when no split is given: its full rank by default (no
# domain), overloaded by a backend that stores a split (a `TensorMap` returns `numout`).
ndims_codomain(a) = ndims(a)

"""
    is_projected(dest, src, ndims_codomain::Val; kwargs...) -> Bool
    is_projected(dest, src; kwargs...) -> Bool

Whether the projected `dest` still represents `src` within the `isapprox` tolerance, i.e. whether
the projection that produced `dest` discarded only a negligible component of `src`. Keyword
arguments are forwarded to `isapprox`. Compares `src` against [`unproject`](@ref)`(dest, ndims_codomain)`, so a backend that changes basis in `project` is checked in the frame `src` was
given in. The two-argument form uses the destination's own codomain rank.

Together with [`unchecked_project`](@ref) this is the backend customization point ([`project`](@ref)
and [`tryproject`](@ref) derive from the two).
"""
function is_projected(dest, src, ndims_codomain::Val; kwargs...)
    check_project_size(size(src), size(dest))
    return isapprox(reshape(src, size(dest)), unproject(dest, ndims_codomain); kwargs...)
end
function is_projected(dest, src; kwargs...)
    return is_projected(dest, src, Val(ndims_codomain(dest)); kwargs...)
end

"""
    unproject(a, ndims_codomain::Val) -> raw

Inverse of [`project`](@ref): recover the dense array that `project` maps to `a`, given the
codomain/domain split `ndims_codomain` as a `Val`. The default is `convert(Array, a)`; a backend
that changes basis in `project` overloads this to undo that change, so that

    unproject(project(raw, codomain_axes, domain_axes), Val(length(codomain_axes))) ≈ raw
"""
unproject(a, ::Val) = convert(Array, a)

"""
    project!(dest, src; kwargs...) -> dest

In-place checked projection: project `src` into the restricted space of
`dest` via [`projectto!`](@ref) and verify with [`is_projected`](@ref) that
only a negligible component was discarded, throwing an `InexactError`
otherwise (keyword arguments are forwarded to the `isapprox` tolerance
check). This is the checked sibling of the [`projectto!`](@ref) primitive,
in the way `copy!` relates to `copyto!`; see [`project`](@ref) for the
allocating form.
"""
function project!(dest, src; kwargs...)
    projectto!(dest, src)
    is_projected(dest, src; kwargs...) ||
        throw(InexactError(:project!, typeof(dest), src))
    return dest
end

"""
    project(raw, codomain_axes, domain_axes; kwargs...) -> dest
    project(raw, axes; kwargs...) -> dest

Project `raw` into a symmetry-restricted array shaped as a map from
`domain_axes` to `codomain_axes`, verifying that only a negligible component
of `raw` is discarded and throwing an `InexactError` otherwise (keyword
arguments are forwarded to the `isapprox` tolerance check; the default
tolerances are subject to change in future versions). See
[`tryproject`](@ref) for a nullable version and [`unchecked_project`](@ref)
for the unchecked projection this derives from.

`raw` must not have more axes than `codomain_axes`/`domain_axes` account for:
`project` projects into exactly the given axes. To append a derived
flux-carrying auxiliary axis (for a charge-shifting operator or a
non-invariant state), use [`project_aux`](@ref). The two-argument form takes a
flat list of `axes` and is equivalent to an empty domain.
"""
function project(raw, codomain_axes, domain_axes; kwargs...)
    dest = unchecked_project(raw, codomain_axes, domain_axes)
    is_projected(dest, raw, Val(length(codomain_axes)); kwargs...) ||
        throw(InexactError(:project, typeof(dest), raw))
    return dest
end
project(raw, axes; kwargs...) = project(raw, axes, (); kwargs...)

"""
    tryproject(raw, codomain_axes, domain_axes; kwargs...) -> Union{dest, Nothing}
    tryproject(raw, axes; kwargs...) -> Union{dest, Nothing}

Like [`project`](@ref), but return `nothing` instead of throwing when more
than a negligible component of `raw` would be discarded. Useful for
branching on whether `raw` is symmetry-allowed in the given axes, e.g.
projecting a state as invariant and falling back to deriving an auxiliary
flux-carrying leg:

    @something tryproject(v, (cod,)) project_aux(v, (cod,))

Keyword arguments are forwarded to the `isapprox` tolerance check.
"""
function tryproject(raw, codomain_axes, domain_axes; kwargs...)
    dest = unchecked_project(raw, codomain_axes, domain_axes)
    return is_projected(dest, raw, Val(length(codomain_axes)); kwargs...) ? dest : nothing
end
tryproject(raw, axes; kwargs...) = tryproject(raw, axes, (); kwargs...)

"""
    infer_aux_space(raw, codomain_axes, domain_axes) -> aux

Derive the auxiliary axis the `*_aux` projection verbs append as the last
domain axis, so the projected result is symmetry-allowed. `raw` carries the
trailing slice axis whose space is derived. This is the backend customization
point for that derivation: the generic (dense) method takes the space straight
from `raw`, while a symmetric backend reads it from the sector structure (a
graded backend derives per-slice sectors, the `TensorMap` backend scans the
`codomain ⊗ conj(domain)` content).
"""
function infer_aux_space(raw, codomain_axes, domain_axes)
    return axes(raw, length(codomain_axes) + length(domain_axes) + 1)
end

# Reshape a physical-rank `raw` up to one trailing slice axis, derive the auxiliary space, and
# return the `(raw, codomain_axes, domain_axes)` triple to forward to a projection verb, with the
# aux appended to the domain. A rank beyond one surplus axis is an error. Shared by the three
# `*_aux` verbs below.
function project_aux_args(raw, codomain_axes, domain_axes)
    nphys = length(codomain_axes) + length(domain_axes)
    nphys <= ndims(raw) <= nphys + 1 || throw(
        ArgumentError(
            "`project_aux` expected a rank-$nphys or rank-$(nphys + 1) input for $nphys given \
            axes, got rank $(ndims(raw))"
        )
    )
    slices = ndims(raw) == nphys ? reshape(raw, (size(raw)..., 1)) : raw
    aux = infer_aux_space(slices, codomain_axes, domain_axes)
    return slices, codomain_axes, (domain_axes..., aux)
end

"""
    project_aux(raw, codomain_axes, domain_axes; kwargs...) -> dest
    project_aux(raw, axes; kwargs...) -> dest

Project `raw` and append a derived auxiliary domain axis carrying its flux,
giving a symmetry-allowed result whose squeezed data is the input (the
flux-canceling MPO-virtual-leg idiom). Unlike [`project`](@ref), which projects
into exactly the given axes, `project_aux` derives the extra leg (see
[`infer_aux_space`](@ref)). `raw` may have the physical rank (a single operator
or state, its flux on a length-1 leg) or one trailing slice axis (an operator
multiplet as laid out by `stack`). Like `project`, it verifies that only a
negligible component is discarded; see [`unchecked_project_aux`](@ref) and
[`tryproject_aux`](@ref) for the unchecked and nullable siblings.
"""
function project_aux(raw, codomain_axes, domain_axes; kwargs...)
    return project(project_aux_args(raw, codomain_axes, domain_axes)...; kwargs...)
end
project_aux(raw, axes; kwargs...) = project_aux(raw, axes, (); kwargs...)

"""
    unchecked_project_aux(raw, codomain_axes, domain_axes) -> dest
    unchecked_project_aux(raw, axes) -> dest

The unchecked sibling of [`project_aux`](@ref): derive and append the auxiliary
axis, then project without verifying which components are discarded.
"""
function unchecked_project_aux(raw, codomain_axes, domain_axes)
    return unchecked_project(project_aux_args(raw, codomain_axes, domain_axes)...)
end
unchecked_project_aux(raw, axes) = unchecked_project_aux(raw, axes, ())

"""
    tryproject_aux(raw, codomain_axes, domain_axes; kwargs...) -> Union{dest, Nothing}
    tryproject_aux(raw, axes; kwargs...) -> Union{dest, Nothing}

The nullable sibling of [`project_aux`](@ref): derive and append the auxiliary
axis, returning `nothing` instead of throwing when more than a negligible
component of `raw` would be discarded.
"""
function tryproject_aux(raw, codomain_axes, domain_axes; kwargs...)
    return tryproject(project_aux_args(raw, codomain_axes, domain_axes)...; kwargs...)
end
tryproject_aux(raw, axes; kwargs...) = tryproject_aux(raw, axes, (); kwargs...)
