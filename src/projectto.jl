# Throw unless `sz1` and `sz2` are equal up to trailing length-1 axes, so a `reshape` cannot
# silently reinterpret same-length data of a different shape.
function check_project_size(sz1::Dims, sz2::Dims)
    all(i -> get(sz1, i, 1) == get(sz2, i, 1), 1:max(length(sz1), length(sz2))) || throw(
        DimensionMismatch("sizes $sz1 and $sz2 differ beyond trailing length-1 axes")
    )
    return nothing
end

"""
    projectto!(dest, src) -> dest

Project `src` into the restricted space of `dest` without checking which
components may have been projected out. The sizes of `src` and `dest` may differ
only by trailing length-1 axes. See [`project!`](@ref) for a checked version.
"""
function projectto!(dest, src)
    check_project_size(size(src), size(dest))
    return copyto!(dest, reshape(src, size(dest)))
end

"""
    allocate_project(raw, axes_codomain, axes_domain) -> dest

Allocate the destination that projecting `raw` onto
`axes_codomain`/`axes_domain` fills.

`project` projects into exactly the given axes, so `raw` must not have more
axes than they account for. To append a derived flux-carrying auxiliary axis
for a charge-shifting operator or a non-invariant state, use
[`project_aux`](@ref) instead.
"""
function allocate_project(raw, axes_codomain, axes_domain)
    nphys = length(axes_codomain) + length(axes_domain)
    ndims(raw) <= nphys || throw(
        ArgumentError(
            "`project` projects into exactly the given axes and does not derive an auxiliary \
            axis; got a rank-$(ndims(raw)) input for $nphys given axes. Use `project_aux` to \
            append a derived flux-carrying leg, or pass the axis explicitly."
        )
    )
    return similar_map(raw, axes_codomain, axes_domain)
end

"""
    unchecked_project(raw, axes_codomain, axes_domain) -> dest
    unchecked_project(raw, axes) -> dest

Project `raw` into a symmetry-restricted array shaped as a map from
`axes_domain` to `axes_codomain`, without checking which components are
discarded: entries of `raw` outside the symmetry-allowed structure are
dropped without inspection. Most callers want [`project`](@ref), which
verifies that nothing was discarded, or [`tryproject`](@ref), which returns
`nothing` instead of throwing. The two-argument form takes a flat list of `axes`
and is equivalent to an empty domain.
"""
function unchecked_project(raw, axes_codomain, axes_domain)
    return projectto!(allocate_project(raw, axes_codomain, axes_domain), raw)
end
unchecked_project(raw, axes) = unchecked_project(raw, axes, ())

"""
    TensorAlgebra.ndims_codomain(a) -> Int
    TensorAlgebra.ndims_domain(a) -> Int

The codomain and domain ranks of `a`'s intrinsic split, for a type that stores one. An array has
no intrinsic split, so it is all codomain with an empty domain. A type that stores a split
overloads `ndims_codomain` and [`has_bipartition`](@ref).
"""
ndims_codomain(a) = ndims(a)
@doc (@doc ndims_codomain)
ndims_domain(a) = ndims(a) - ndims_codomain(a)

"""
    TensorAlgebra.has_bipartition(a) -> Bool

Whether `a`'s type stores an intrinsic codomain/domain split, so that
[`ndims_codomain`](@ref) reports a split `a` genuinely carries rather than the all-codomain
fallback. A `TensorMap` returns `true` and an array returns `false`.
"""
has_bipartition(a) = false

"""
    is_projected(dest, src, ndims_codomain::Val; kwargs...) -> Bool
    is_projected(dest, src; kwargs...) -> Bool

Whether the projected `dest` still represents `src` within the `isapprox` tolerance, that is,
whether the projection that produced `dest` discarded only a negligible component of `src`.
`src` is compared against [`unproject`](@ref)`(dest, ndims_codomain)`, and keyword arguments are
forwarded to `isapprox`. The two-argument form uses the codomain rank of `dest`.
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
codomain/domain split `ndims_codomain` as a `Val`, so that

    unproject(project(raw, axes_codomain, axes_domain), Val(length(axes_codomain))) ≈ raw
"""
unproject(a, ::Val) = convert(Array, a)

"""
    project!(dest, src; kwargs...) -> dest

In-place checked projection: project `src` into the restricted space of
`dest` and verify that only a negligible component was discarded, throwing an
`InexactError` otherwise. Keyword arguments are forwarded to the `isapprox`
tolerance check. See [`projectto!`](@ref) for the unchecked version and
[`project`](@ref) for the allocating form.
"""
function project!(dest, src; kwargs...)
    projectto!(dest, src)
    is_projected(dest, src; kwargs...) ||
        throw(InexactError(:project!, typeof(dest), src))
    return dest
end

"""
    project(raw, axes_codomain, axes_domain; kwargs...) -> dest
    project(raw, axes; kwargs...) -> dest

Project `raw` into a symmetry-restricted array shaped as a map from
`axes_domain` to `axes_codomain`, verifying that only a negligible component
of `raw` is discarded and throwing an `InexactError` otherwise. Keyword
arguments are forwarded to the `isapprox` tolerance check, and the default
tolerances are subject to change in future versions. See [`tryproject`](@ref)
for a nullable version and [`unchecked_project`](@ref) for an unchecked version.

`raw` must not have more axes than `axes_codomain`/`axes_domain` account for:
`project` projects into exactly the given axes. To append a derived
flux-carrying auxiliary axis (for a charge-shifting operator or a
non-invariant state), use [`project_aux`](@ref). The two-argument form takes a
flat list of `axes` and is equivalent to an empty domain.
"""
function project(raw, axes_codomain, axes_domain; kwargs...)
    dest = unchecked_project(raw, axes_codomain, axes_domain)
    is_projected(dest, raw, Val(length(axes_codomain)); kwargs...) ||
        throw(InexactError(:project, typeof(dest), raw))
    return dest
end
project(raw, axes; kwargs...) = project(raw, axes, (); kwargs...)

"""
    tryproject(raw, axes_codomain, axes_domain; kwargs...) -> Union{dest, Nothing}
    tryproject(raw, axes; kwargs...) -> Union{dest, Nothing}

Like [`project`](@ref), but return `nothing` instead of throwing when more
than a negligible component of `raw` would be discarded. Useful for
branching on whether `raw` is symmetry-allowed in the given axes, e.g.
projecting a state as invariant and falling back to deriving an auxiliary
flux-carrying leg:

    @something tryproject(v, (cod,)) project_aux(v, (cod,))

Keyword arguments are forwarded to the `isapprox` tolerance check.
"""
function tryproject(raw, axes_codomain, axes_domain; kwargs...)
    dest = unchecked_project(raw, axes_codomain, axes_domain)
    return is_projected(dest, raw, Val(length(axes_codomain)); kwargs...) ? dest : nothing
end
tryproject(raw, axes; kwargs...) = tryproject(raw, axes, (); kwargs...)

"""
    infer_aux_space(raw, axes_codomain, axes_domain) -> aux

Derive the auxiliary axis the `*_aux` projection verbs append as the last
domain axis, so the projected result is symmetry-allowed. `raw` carries the
trailing slice axis whose space is derived.
"""
function infer_aux_space(raw, axes_codomain, axes_domain)
    return axes(raw, length(axes_codomain) + length(axes_domain) + 1)
end

# The arguments to forward to a projection verb, with `raw` given one trailing slice axis and the
# derived aux axis appended to the domain.
function project_aux_args(raw, axes_codomain, axes_domain)
    nphys = length(axes_codomain) + length(axes_domain)
    nphys <= ndims(raw) <= nphys + 1 || throw(
        ArgumentError(
            "`project_aux` expected a rank-$nphys or rank-$(nphys + 1) input for $nphys given \
            axes, got rank $(ndims(raw))"
        )
    )
    slices = ndims(raw) == nphys ? reshape(raw, (size(raw)..., 1)) : raw
    aux = infer_aux_space(slices, axes_codomain, axes_domain)
    return slices, axes_codomain, (axes_domain..., aux)
end

"""
    project_aux(raw, axes_codomain, axes_domain; kwargs...) -> dest
    project_aux(raw, axes; kwargs...) -> dest

Project `raw` and append a derived auxiliary domain axis carrying its flux,
giving a symmetry-allowed result whose squeezed data is the input (the
flux-canceling MPO-virtual-leg idiom). Unlike [`project`](@ref), which projects
into exactly the given axes, `project_aux` derives the extra leg (see
[`infer_aux_space`](@ref)). `raw` may have the physical rank (a single operator
or state, its flux on a length-1 leg) or one trailing slice axis (an operator
multiplet as laid out by `stack`). Like `project`, it verifies that only a
negligible component is discarded. See [`unchecked_project_aux`](@ref) and
[`tryproject_aux`](@ref) for unchecked and nullable versions.
"""
function project_aux(raw, axes_codomain, axes_domain; kwargs...)
    return project(project_aux_args(raw, axes_codomain, axes_domain)...; kwargs...)
end
project_aux(raw, axes; kwargs...) = project_aux(raw, axes, (); kwargs...)

"""
    unchecked_project_aux(raw, axes_codomain, axes_domain) -> dest
    unchecked_project_aux(raw, axes) -> dest

The unchecked sibling of [`project_aux`](@ref): derive and append the auxiliary
axis, then project without verifying which components are discarded.
"""
function unchecked_project_aux(raw, axes_codomain, axes_domain)
    return unchecked_project(project_aux_args(raw, axes_codomain, axes_domain)...)
end
unchecked_project_aux(raw, axes) = unchecked_project_aux(raw, axes, ())

"""
    tryproject_aux(raw, axes_codomain, axes_domain; kwargs...) -> Union{dest, Nothing}
    tryproject_aux(raw, axes; kwargs...) -> Union{dest, Nothing}

The nullable sibling of [`project_aux`](@ref): derive and append the auxiliary
axis, returning `nothing` instead of throwing when more than a negligible
component of `raw` would be discarded.
"""
function tryproject_aux(raw, axes_codomain, axes_domain; kwargs...)
    return tryproject(project_aux_args(raw, axes_codomain, axes_domain)...; kwargs...)
end
tryproject_aux(raw, axes; kwargs...) = tryproject_aux(raw, axes, (); kwargs...)
