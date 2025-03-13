using MatrixAlgebraKit:
  eig_full,
  eig_trunc,
  eig_vals,
  eigh_full,
  eigh_trunc,
  eigh_vals,
  lq_full,
  lq_compact,
  qr_full,
  qr_compact,
  svd_full,
  svd_compact,
  svd_trunc,
  svd_vals
using LinearAlgebra: LinearAlgebra

# TODO: consider in-place version
# TODO: figure out kwargs and document
#
"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; full=false, kwargs...) -> Q, R
    qr(A::AbstractArray, biperm::BlockedPermutation{2}; full=false, kwargs...) -> Q, R

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return qr(A, biperm)
end
function qr(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=false, kwargs...)
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

"""
    lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; full=false, kwargs...) -> L, Q
    lq(A::AbstractArray, biperm::BlockedPermutation{2}; full=false, kwargs...) -> L, Q

Compute the LQ decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
"""
function lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return lq(A, biperm)
end
function lq(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=false, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  L, Q = full ? lq_full(A_mat; kwargs...) : lq_compact(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_L = (axes_codomain..., axes(L, ndims(L)))
  axes_Q = (axes(Q, 1), axes_domain...)
  return splitdims(L, axes_L), splitdims(Q, axes_Q)
end

# TODO: what name do we want?
"""
    eig(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eig(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `ishermitian::Bool`: specify if the matrix is Hermitian, which can be used to speed up the
    computation. If `false`, the output `eltype` will always be `<:Complex`.
- `trunc`: Truncation keywords for `eig(h)_trunc`.
- Other keywords are passed on directly to MatrixAlgebraKit
"""
function eig(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return eig(A, biperm; kwargs...)
end
function eig(
  A::AbstractArray,
  biperm::BlockedPermutation{2};
  trunc=nothing,
  ishermitian=nothing,
  kwargs...,
)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  ishermitian = @something ishermitian LinearAlgebra.ishermitian(A_mat)

  # factorization
  if !isnothing(trunc)
    D, V = (ishermitian ? eigh_trunc : eig_trunc)(A_mat; trunc, kwargs...)
  else
    D, V = (ishermitian ? eigh_full : eig_full)(A_mat; kwargs...)
  end

  # matrix to tensor
  axes_codomain, = blockpermute(axes(A), biperm)
  axes_V = (axes_codomain..., axes(V, ndims(V)))
  return D, splitdims(V, axes_V)
end

"""
    eigvals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eigvals(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`. The output is a vector of eigenvalues.

## Keyword arguments

- `ishermitian::Bool`: specify if the matrix is Hermitian, which can be used to speed up the
    computation. If `false`, the output `eltype` will always be `<:Complex`.
- Other keywords are passed on directly to MatrixAlgebraKit
"""
function eigvals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return eigvals(A, biperm; kwargs...)
end
function eigvals(
  A::AbstractArray, biperm::BlockedPermutation{2}; ishermitian=nothing, kwargs...
)
  A_mat = fusedims(A, biperm)
  ishermitian = @something ishermitian LinearAlgebra.ishermitian(A_mat)
  return (ishermitian ? eigh_vals : eig_vals)(A_mat; kwargs...)
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

"""
    svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain) -> S
    svdvals(A::AbstractArray, biperm::BlockedPermutation{2}) -> S

Compute the singular values of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`. The output is a vector of singular values.
"""
function svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return svdvals(A, biperm)
end
function svdvals(A::AbstractArray, biperm::BlockedPermutation{2})
  A_mat = fusedims(A, biperm)
  return svd_vals(A_mat)
end
