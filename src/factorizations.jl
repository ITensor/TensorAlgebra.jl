using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit

for f in (
        :qr, :lq, :left_polar, :right_polar, :polar, :left_orth, :right_orth, :orth, :factorize,
    )
    @eval begin
        function $f(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
            biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(A, biperm; kwargs...)
        end
        function $f(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
            # tensor to matrix
            A_mat = matricize(A, biperm)

            # factorization
            X, Y = MatrixAlgebra.$f(A_mat; kwargs...)

            # matrix to tensor
            axes_codomain, axes_domain = blocks(axes(A)[biperm])
            axes_X = tuplemortar((axes_codomain, (axes(X, 2),)))
            axes_Y = tuplemortar(((axes(Y, 1),), axes_domain))
            return unmatricize(X, axes_X), unmatricize(Y, axes_Y)
        end
    end
end

"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> Q, R

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `R` is square, respectively.
- `positive::Bool=false`: specify if the diagonal of `R` should be positive, leading to a unique decomposition.
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.qr_full!` and `MatrixAlgebraKit.qr_compact!`.
"""
qr

"""
    lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> L, Q

Compute the LQ decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `L` is square, respectively.
- `positive::Bool=false`: specify if the diagonal of `L` should be positive, leading to a unique decomposition.
- Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.lq_full!` and `MatrixAlgebraKit.lq_compact!`.
"""
lq

"""
    left_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> W, P
    left_polar(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> W, P

Compute the left polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_polar!`.
"""
left_polar

"""
    right_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> P, W
    right_polar(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> P, W

Compute the right polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_polar!`.
"""
right_polar

"""
    left_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> V, C
    left_orth(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> V, C

Compute the left orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_orth!`.
"""
left_orth

"""
    right_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> C, V
    right_orth(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> C, V

Compute the right orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_orth!`.
"""
right_orth

"""
    factorize(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> X, Y
    factorize(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> X, Y

Compute the decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`.

## Keyword arguments

- `orth::Symbol=:left`: specify the orthogonality of the decomposition.
  Currently only `:left` and `:right` are supported.
- Other keywords are passed on directly to MatrixAlgebraKit.
"""
factorize

"""
    eigen(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eigen(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> D, V

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
function eigen(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
    # tensor to matrix
    A_mat = matricize(A, biperm)

    # factorization
    D, V = MatrixAlgebra.eigen!(A_mat; kwargs...)

    # matrix to tensor
    axes_codomain, = blocks(axes(A)[biperm])
    axes_V = tuplemortar((axes_codomain, (axes(V, ndims(V)),)))
    return D, unmatricize(V, axes_V)
end

"""
    eigvals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eigvals(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> D

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
function eigvals(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
    A_mat = matricize(A, biperm)
    return MatrixAlgebra.eigvals!(A_mat; kwargs...)
end

"""
    svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> U, S, Vᴴ

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
function svd(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
    # tensor to matrix
    A_mat = matricize(A, biperm)

    # factorization
    U, S, Vᴴ = MatrixAlgebra.svd!(A_mat; kwargs...)

    # matrix to tensor
    axes_codomain, axes_domain = blocks(axes(A)[biperm])
    axes_U = tuplemortar((axes_codomain, (axes(U, 2),)))
    axes_Vᴴ = tuplemortar(((axes(Vᴴ, 1),), axes_domain))
    return unmatricize(U, axes_U), S, unmatricize(Vᴴ, axes_Vᴴ)
end

"""
    svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain) -> S
    svdvals(A::AbstractArray, biperm::AbstractBlockPermutation{2}) -> S

Compute the singular values of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain indices. These can be specified either via
their labels, or directly through a `biperm`. The output is a vector of singular values.

See also `MatrixAlgebraKit.svd_vals!`.
"""
function svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain)
    biperm = blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
    return svdvals(A, biperm)
end
function svdvals(A::AbstractArray, biperm::AbstractBlockPermutation{2})
    A_mat = matricize(A, biperm)
    return MatrixAlgebra.svdvals!(A_mat)
end

"""
    left_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> N
    left_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> N

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
function left_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
    A_mat = matricize(A, biperm)
    N = MatrixAlgebraKit.left_null!(A_mat; kwargs...)
    axes_codomain = first(blocks(axes(A)[biperm]))
    axes_N = tuplemortar((axes_codomain, (axes(N, 2),)))
    return unmatricize(N, axes_N)
end

"""
    right_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Nᴴ
    right_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> Nᴴ

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
function right_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
    A_mat = matricize(A, biperm)
    Nᴴ = MatrixAlgebraKit.right_null!(A_mat; kwargs...)
    axes_domain = last(blocks((axes(A)[biperm])))
    axes_Nᴴ = tuplemortar(((axes(Nᴴ, 1),), axes_domain))
    return unmatricize(Nᴴ, axes_Nᴴ)
end
