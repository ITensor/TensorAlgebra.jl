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
`codomain_axes`/`domain_axes` fills. The generic method is
`similar_map(raw, codomain_axes, domain_axes)`. This is a backend
customization point (with [`projectto!`](@ref) and [`is_projected`](@ref)):
the allocation may depend on the data, since a symmetric backend derives the
space of one trailing surplus axis in `raw` (an auxiliary leg appended as
the last domain axis, e.g. a flux-canceling leg for a charge-shifting
operator) before allocating.
"""
function allocate_project(raw, codomain_axes, domain_axes)
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
# Forward to the three-argument form so a backend's surplus-axis derivation also
# applies to the flat all-codomain (state) form.
unchecked_project(raw, axes) = unchecked_project(raw, axes, ())

"""
    is_projected(dest, src; kwargs...) -> Bool

Whether the projected `dest` still represents `src` within the `isapprox`
tolerance, i.e. whether the projection that produced `dest` discarded only a
negligible component of `src`. Keyword arguments are forwarded to `isapprox`.

Together with [`unchecked_project`](@ref) this is the backend customization
point ([`project`](@ref) and [`tryproject`](@ref) derive from the two). The
generic method reshapes `src` to `size(dest)` up to trailing length-1 axes
(so a lower-rank `src` that omits them lines up) and compares against
`convert(Array, dest)`, so a backend whose arrays are not elementwise
comparable to a dense array (opaque block storage, a `TensorMap`) only needs
that conversion.
"""
function is_projected(dest, src; kwargs...)
    check_project_size(size(src), size(dest))
    return isapprox(reshape(src, size(dest)), convert(Array, dest); kwargs...)
end

# Compare `src` against `unproject(dest, ndims_codomain)` rather than `convert(Array, dest)`, so a
# backend that changes basis in `project` is checked in the frame `raw` was given in.
function is_projected(dest, src, ndims_codomain::Val; kwargs...)
    check_project_size(size(src), size(dest))
    return isapprox(reshape(src, size(dest)), unproject(dest, ndims_codomain); kwargs...)
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

When `raw` has one axis more than the given axes account for, that trailing
surplus axis is an auxiliary leg whose space a symmetric backend derives so
the result is symmetry-allowed (e.g. a flux-canceling leg for a
charge-shifting operator); the result's shape matches `raw`'s shape. The
derivation is backend-internal: a graded backend reads the sector, the
`TensorMap` backend projects over the `codomain ⊗ conj(domain)` content. The
two-argument form takes a flat list of `axes` and is equivalent to an empty
domain.
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

    @something tryproject(v, (cod,)) project(reshape(v, (length(v), 1)), (cod,))

Keyword arguments are forwarded to the `isapprox` tolerance check.
"""
function tryproject(raw, codomain_axes, domain_axes; kwargs...)
    dest = unchecked_project(raw, codomain_axes, domain_axes)
    return is_projected(dest, raw, Val(length(codomain_axes)); kwargs...) ? dest : nothing
end
tryproject(raw, axes; kwargs...) = tryproject(raw, axes, (); kwargs...)
