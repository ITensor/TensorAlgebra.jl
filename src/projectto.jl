"""
    projectto!(dest, src) -> dest

Project `src` into the restricted space of `dest` without checking which
components may have been projected out. Defaults to `copyto!`. See
[`checked_projectto!`](@ref) for a checked version.
"""
projectto!(dest, src) = copyto!(dest, src)

"""
    checked_projectto!(dest, src; kwargs...) -> dest

Project `src` into the restricted space of `dest` via [`projectto!`](@ref)
and verify via `isapprox(src, dest; kwargs...)` that the discarded
component is within tolerance. Keyword arguments are forwarded to
`isapprox`.
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
into it with [`projectto!`](@ref). See [`checked_project_map`](@ref) for
a checked version.
"""
function project_map(raw, codomain_axes, domain_axes)
    return projectto!(similar_map(raw, eltype(raw), codomain_axes, domain_axes), raw)
end

"""
    checked_project_map(raw, codomain_axes, domain_axes; kwargs...) -> dest

Allocate a map-shaped array via [`similar_map`](@ref) and project `raw`
into it with [`checked_projectto!`](@ref). Keyword arguments are forwarded
to [`checked_projectto!`](@ref).
"""
function checked_project_map(raw, codomain_axes, domain_axes; kwargs...)
    return checked_projectto!(
        similar_map(raw, eltype(raw), codomain_axes, domain_axes), raw; kwargs...
    )
end
