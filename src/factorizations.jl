using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit:
  eig_full!,
  eig_trunc!,
  eig_vals!,
  eigh_full!,
  eigh_trunc!,
  eigh_vals!,
  left_null!,
  left_orth!,
  left_polar!,
  lq_full!,
  lq_compact!,
  qr_full!,
  qr_compact!,
  right_null!,
  right_orth!,
  right_polar!,
  svd_full!,
  svd_compact!,
  svd_trunc!,
  svd_vals!

"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> Q, R

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `R` is square, respectively.
- `positive::Bool=false`: specify if the diagonal of `R` should be positive, leading to a unique decomposition.
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.qr_full!` and `MatrixAlgebraKit.qr_compact!`.
"""
function qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return qr(A, biperm; kwargs...)
end
function qr(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=false, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  Q, R = full ? qr_full!(A_mat; kwargs...) : qr_compact!(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_Q = (axes_codomain..., axes(Q, 2))
  axes_R = (axes(R, 1), axes_domain...)
  return splitdims(Q, axes_Q), splitdims(R, axes_R)
end

"""
    lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> L, Q

Compute the LQ decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `L` is square, respectively.
- `positive::Bool=false`: specify if the diagonal of `L` should be positive, leading to a unique decomposition.
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.lq_full!` and `MatrixAlgebraKit.lq_compact!`.
"""
function lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return lq(A, biperm; kwargs...)
end
function lq(A::AbstractArray, biperm::BlockedPermutation{2}; full::Bool=false, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  L, Q = (full ? lq_full! : lq_compact!)(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_L = (axes_codomain..., axes(L, ndims(L)))
  axes_Q = (axes(Q, 1), axes_domain...)
  return splitdims(L, axes_L), splitdims(Q, axes_Q)
end

"""
    eigen(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eigen(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `ishermitian::Bool`: specify if the matrix is Hermitian, which can be used to speed up the
    computation. If `false`, the output `eltype` will always be `<:Complex`.
- `trunc`: Truncation keywords for `eig(h)_trunc`.
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.eig_full!`, `MatrixAlgebraKit.eig_trunc!`, `MatrixAlgebraKit.eig_vals!`,
`MatrixAlgebraKit.eigh_full!`, `MatrixAlgebraKit.eigh_trunc!`, and `MatrixAlgebraKit.eigh_vals!`.
"""
function eigen(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return eigen(A, biperm; kwargs...)
end
function eigen(
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
  f! = if !isnothing(trunc)
    ishermitian ? eigh_trunc! : eig_trunc!
  else
    ishermitian ? eigh_full! : eig_full!
  end
  D, V = f!(A_mat; kwargs...)

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
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.eig_vals!` and `MatrixAlgebraKit.eigh_vals!`.
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
  return (ishermitian ? eigh_vals! : eig_vals!)(A_mat; kwargs...)
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
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.svd_full!`, `MatrixAlgebraKit.svd_compact!`, and `MatrixAlgebraKit.svd_trunc!`.
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
    U, S, Vᴴ = svd_trunc!(A_mat; trunc, kwargs...)
  else
    U, S, Vᴴ = full ? svd_full!(A_mat; kwargs...) : svd_compact!(A_mat; kwargs...)
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

See also `MatrixAlgebraKit.svd_vals!`.
"""
function svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return svdvals(A, biperm)
end
function svdvals(A::AbstractArray, biperm::BlockedPermutation{2})
  A_mat = fusedims(A, biperm)
  return svd_vals!(A_mat)
end

"""
    left_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> N
    left_null(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> N

Compute the left nullspace of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
The output satisfies `N' * A ≈ 0` and `N' * N ≈ I`.

## Keyword arguments

- `atol::Real=0`: absolute tolerance for the nullspace computation.
- `rtol::Real=0`: relative tolerance for the nullspace computation.
- `kind::Symbol`: specify the kind of decomposition used to compute the nullspace.
  The options are `:qr`, `:qrpos` and `:svd`. The former two require `0 == atol == rtol`.
  The default is `:qrpos` if `atol == rtol == 0`, and `:svd` otherwise.
"""
function left_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return left_null(A, biperm; kwargs...)
end
function left_null(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  A_mat = fusedims(A, biperm)
  N = left_null!(A_mat; kwargs...)
  axes_codomain, _ = blockpermute(axes(A), biperm)
  axes_N = (axes_codomain..., axes(N, 2))
  N_tensor = splitdims(N, axes_N)
  return N_tensor
end

"""
    right_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Nᴴ
    right_null(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> Nᴴ

Compute the right nullspace of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.
The output satisfies `A * Nᴴ' ≈ 0` and `Nᴴ * Nᴴ' ≈ I`.

## Keyword arguments

- `atol::Real=0`: absolute tolerance for the nullspace computation.
- `rtol::Real=0`: relative tolerance for the nullspace computation.
- `kind::Symbol`: specify the kind of decomposition used to compute the nullspace.
  The options are `:lq`, `:lqpos` and `:svd`. The former two require `0 == atol == rtol`.
  The default is `:lqpos` if `atol == rtol == 0`, and `:svd` otherwise.
"""
function right_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return right_null(A, biperm; kwargs...)
end
function right_null(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  A_mat = fusedims(A, biperm)
  Nᴴ = right_null!(A_mat; kwargs...)
  _, axes_domain = blockpermute(axes(A), biperm)
  axes_Nᴴ = (axes(Nᴴ, 1), axes_domain...)
  return splitdims(Nᴴ, axes_Nᴴ)
end

"""
    left_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> W, P
    left_polar(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> W, P

Compute the left polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_polar!`.
"""
function left_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return left_polar(A, biperm; kwargs...)
end
function left_polar(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  W, P = left_polar!(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_W = (axes_codomain..., axes(W, 2))
  axes_P = (axes(P, 1), axes_domain...)
  return splitdims(W, axes_W), splitdims(P, axes_P)
end

"""
    right_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> P, W
    right_polar(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> P, W

Compute the right polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_polar!`.
"""
function right_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return right_polar(A, biperm; kwargs...)
end
function right_polar(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  P, W = right_polar!(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_P = (axes_codomain..., axes(P, ndims(P)))
  axes_W = (axes(W, 1), axes_domain...)
  return splitdims(P, axes_P), splitdims(W, axes_W)
end

"""
    left_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> V, C
    left_orth(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> V, C

Compute the left orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_orth!`.
"""
function left_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return left_orth(A, biperm; kwargs...)
end
function left_orth(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  V, C = left_orth!(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_V = (axes_codomain..., axes(V, 2))
  axes_C = (axes(C, 1), axes_domain...)
  return splitdims(V, axes_V), splitdims(C, axes_C)
end

"""
    right_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> C, V
    right_orth(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> C, V

Compute the right orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_orth!`.
"""
function right_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return right_orth(A, biperm; kwargs...)
end
function right_orth(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  P, W = right_orth!(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_P = (axes_codomain..., axes(P, ndims(P)))
  axes_W = (axes(W, 1), axes_domain...)
  return splitdims(P, axes_P), splitdims(W, axes_W)
end

"""
    factorize(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> X, Y
    factorize(A::AbstractArray, biperm::BlockedPermutation{2}; kwargs...) -> X, Y

Compute the decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `orth::Symbol=:left`: specify the orthogonality of the decomposition.
  Currently only `:left` and `:right` are supported.
- Other keywords are passed on directly to MatrixAlgebraKit.
"""
function factorize(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
  biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
  return factorize(A, biperm; kwargs...)
end
function factorize(A::AbstractArray, biperm::BlockedPermutation{2}; orth=:left, kwargs...)
  # tensor to matrix
  A_mat = fusedims(A, biperm)

  # factorization
  X, Y = (orth == :left ? left_orth! : right_orth!)(A_mat; kwargs...)

  # matrix to tensor
  axes_codomain, axes_domain = blockpermute(axes(A), biperm)
  axes_X = (axes_codomain..., axes(X, ndims(X)))
  axes_Y = (axes(Y, 1), axes_domain...)
  return splitdims(X, axes_X), splitdims(Y, axes_Y)
end
