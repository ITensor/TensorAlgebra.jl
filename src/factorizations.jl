using MatrixAlgebraKit: qr_full, qr_compact, svd_full, svd_compact, svd_trunc

# TODO: consider in-place version
# TODO: figure out kwargs and document
#
"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; full=true, kwargs...) -> Q, R
    qr(A::AbstractArray, biperm::BlockedPermutation{2}; full=true, kwargs...) -> Q, R

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return qr(A, biperm)
end
function qr(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=true, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  Q, R = full ? qr_full(A_mat; kwargs...) : qr_compact(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_Q = (axes_codomain..., axes(Q, 2))
  axes_R = (axes(R, 1), axes_domain...)
  return splitdims(Q, axes_Q), splitdims(R, axes_R)
end

# TODO: separate out the algorithm selection step from the implementation
"""
    svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> U, S, Vᴴ

Compute the SVD decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments
- `full::Bool=false`: select between a "thick" or a "thin" decomposition, where both `U` and `Vᴴ`
  are unitary or isometric.
- `trunc`: Truncation keywords for `svd_trunc`. Not compatible with `full=true`.
- Other keywords are passed on directly to MatrixAlgebraKit
"""
function svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return svd(A, biperm; kwargs...)
end
function svd(
  A::AbstractArray,
  biperm::BlockedPermutation{2};
  full::Bool=false,
  trunc=nothing,
  kwargs...,
)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  if !isnothing(trunc)
    @assert !full "Specified both full and truncation, currently not supported"
    U, S, Vᴴ = svd_trunc(A_mat; trunc, kwargs...)
  else
    U, S, Vᴴ = full ? svd_full(A_mat; kwargs...) : svd_compact(A_mat; kwargs...)
  end

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_U = (axes_codomain..., axes(U, 2))
  axes_Vᴴ = (axes(Vᴴ, 1), axes_domain...)
  return splitdims(U, axes_U), S, splitdims(Vᴴ, axes_Vᴴ)
end
