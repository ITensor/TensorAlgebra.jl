"""
    projectto!(dest, src) -> dest

Orthogonally project `src` onto the representable subspace of `dest`, in place.
Per-backend primitive. Tolerance-free and magnitude-blind: drops the
non-representable component regardless of its size. Pair with
[`checked_projectto!`](@ref) (or `isapprox`-after) when the discarded weight
needs to be verified.

The default falls through to `copyto!`, which is the right behavior for dense
backends where the representable subspace is everything. Block-structured
backends (e.g. `AbelianGradedArray`, `FusionTensor`) overload this to copy only
the symmetry-allowed entries.
"""
projectto!(dest, src) = copyto!(dest, src)

"""
    checked_projectto!(dest, src; atol=0, rtol=…) -> dest

Project `src` into `dest` via [`projectto!`](@ref), then verify that the
discarded component is within the requested tolerance via `isapprox(src, dest; atol, rtol)`. Throws `InexactError` on failure. Backends may specialize this
verb for a fused, cheaper check (e.g. a one-pass norm comparison).
"""
function checked_projectto!(
        dest, src;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(real(eltype(src)))
    )
    projectto!(dest, src)
    isapprox(src, dest; atol, rtol) ||
        throw(InexactError(:checked_projectto!, typeof(dest), src))
    return dest
end
