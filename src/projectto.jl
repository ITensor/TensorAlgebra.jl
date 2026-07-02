"""
    projectto!(dest, src) -> dest

Project `src` into the restricted space of `dest` without checking which
components may have been projected out. Defaults to `copyto!`. See
[`checked_projectto!`](@ref) for a checked version, and [`project`](@ref)
for the allocating form.
"""
projectto!(dest, src) = copyto!(dest, src)

"""
    checked_projectto!(dest, src; kwargs...) -> dest

Project `src` into the restricted space of `dest` via [`projectto!`](@ref)
and verify via `isapprox(src, dest; kwargs...)` that the discarded
component is within tolerance. Keyword arguments are forwarded to
`isapprox`. The default tolerances are subject to change in future
versions.
"""
function checked_projectto!(dest, src; kwargs...)
    projectto!(dest, src)
    isapprox(src, dest; kwargs...) ||
        throw(InexactError(:checked_projectto!, typeof(dest), src))
    return dest
end

"""
    project_map(raw, codomain_axes, domain_axes) -> dest

Allocate a map-shaped array via [`similar_map`](@ref) and project `raw`
into it with [`projectto!`](@ref). This is the strict form that takes an
explicit codomain/domain split; [`project`](@ref) is the convenience entry
point that also accepts a flat list of axes. See
[`checked_project_map`](@ref) for a checked version.
"""
function project_map(raw, codomain_axes, domain_axes)
    return projectto!(similar_map(raw, codomain_axes, domain_axes), raw)
end

"""
    checked_project_map(raw, codomain_axes, domain_axes; kwargs...) -> dest

Allocate a map-shaped array via [`similar_map`](@ref) and project `raw`
into it with [`checked_projectto!`](@ref). Keyword arguments are forwarded
to [`checked_projectto!`](@ref).
"""
function checked_project_map(raw, codomain_axes, domain_axes; kwargs...)
    return checked_projectto!(
        similar_map(raw, codomain_axes, domain_axes), raw; kwargs...
    )
end

"""
    project(raw, codomain_axes, domain_axes) -> dest
    project(raw, axes) -> dest

Project `raw` into a symmetry-restricted array. The three-argument form
takes an explicit codomain/domain split; the two-argument form takes a
flat list of `axes` and is equivalent to an empty domain. Both forward to
[`project_map`](@ref). See [`checked_project`](@ref) for a checked version.
"""
project(raw, codomain_axes, domain_axes) = project_map(raw, codomain_axes, domain_axes)
project(raw, axes) = project_map(raw, axes, ())

"""
    checked_project(raw, codomain_axes, domain_axes; kwargs...) -> dest
    checked_project(raw, axes; kwargs...) -> dest

Checked form of [`project`](@ref): projects `raw` via
[`checked_project_map`](@ref), verifying that the discarded component is
within tolerance. Keyword arguments are forwarded to
[`checked_project_map`](@ref).
"""
function checked_project(raw, codomain_axes, domain_axes; kwargs...)
    return checked_project_map(raw, codomain_axes, domain_axes; kwargs...)
end
function checked_project(raw, axes; kwargs...)
    return checked_project_map(raw, axes, (); kwargs...)
end
