using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit

for (f, f_mat) in (
        (:qr, :(MatrixAlgebra.qr)),
        (:lq, :(MatrixAlgebra.lq)),
        (:left_polar, :(MatrixAlgebraKit.left_polar)),
        (:right_polar, :(MatrixAlgebraKit.right_polar)),
        (:polar, :(MatrixAlgebra.polar)),
        (:left_orth, :(MatrixAlgebraKit.left_orth)),
        (:right_orth, :(MatrixAlgebraKit.right_orth)),
        (:orth, :(MatrixAlgebra.orth)),
        (:factorize, :(MatrixAlgebra.factorize)),
    )
    @eval begin
        function $f(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            X, Y = $f_mat(A_mat; kwargs...)
            biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
            axes_codomain, axes_domain = blocks(blockpermute(axes(A), biperm))
            axes_X = tuplemortar((axes_codomain, (axes(X, 2),)))
            axes_Y = tuplemortar(((axes(Y, 1),), axes_domain))
            return unmatricize(style, X, axes_X), unmatricize(style, Y, axes_Y)
        end
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

for f in (
        :qr, :lq, :left_polar, :right_polar, :polar, :left_orth, :right_orth, :orth,
        :factorize, :eigen, :eigvals, :svd, :svdvals, :left_null, :right_null,
        :gram_eigh_full, :gram_eigh_full_with_pinv, :one,
    )
    @eval begin
        function $f(
                style::FusionStyle, A::AbstractArray,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            A_perm = bipermutedims(A, perm_codomain, perm_domain)
            return $f(style, A_perm, Val(length(perm_codomain)); kwargs...)
        end
        function $f(
                A::AbstractArray,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            A_perm = bipermutedims(A, perm_codomain, perm_domain)
            return $f(A_perm, Val(length(perm_codomain)); kwargs...)
        end

        function $f(
                style::FusionStyle, A::AbstractArray,
                labels_A, labels_codomain, labels_domain; kwargs...
            )
            biperm =
                blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(style, A, blocks(biperm)...; kwargs...)
        end
        function $f(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
            biperm =
                blockedperm_indexin(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(A, blocks(biperm)...; kwargs...)
        end

        function $f(
                style::FusionStyle, A::AbstractArray,
                biperm::AbstractBlockPermutation{2}; kwargs...
            )
            return $f(style, A, blocks(biperm)...; kwargs...)
        end
        function $f(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
            return $f(A, blocks(biperm)...; kwargs...)
        end
    end
end

"""
    qr(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Q, R
    qr(A::AbstractArray, ndims_codomain::Val; kwargs...) -> Q, R
    qr(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> Q, R

Compute the QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `R` is square, respectively.
  - `positive::Bool=false`: specify if the diagonal of `R` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.qr_full!` and `MatrixAlgebraKit.qr_compact!`.
"""
qr

"""
    lq(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> L, Q
    lq(A::AbstractArray, ndims_codomain::Val; kwargs...) -> L, Q
    lq(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> L, Q

Compute the LQ decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - `full::Bool=false`: select between a "full" or a "compact" decomposition, where `Q` is unitary or `L` is square, respectively.
  - `positive::Bool=false`: specify if the diagonal of `L` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.lq_full!` and `MatrixAlgebraKit.lq_compact!`.
"""
lq

"""
    left_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> W, P
    left_polar(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> W, P
    left_polar(A::AbstractArray, ndims_codomain::Val; kwargs...) -> W, P
    left_polar(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> W, P

Compute the left polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_polar!`.
"""
left_polar

"""
    right_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> P, W
    right_polar(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> P, W
    right_polar(A::AbstractArray, ndims_codomain::Val; kwargs...) -> P, W
    right_polar(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> P, W

Compute the right polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_polar!`.
"""
right_polar

"""
    left_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> V, C
    left_orth(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> V, C
    left_orth(A::AbstractArray, ndims_codomain::Val; kwargs...) -> V, C
    left_orth(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> V, C

Compute the left orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_orth!`.
"""
left_orth

"""
    right_orth(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> C, V
    right_orth(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> C, V
    right_orth(A::AbstractArray, ndims_codomain::Val; kwargs...) -> C, V
    right_orth(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> C, V

Compute the right orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_orth!`.
"""
right_orth

"""
    factorize(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> X, Y
    factorize(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> X, Y
    factorize(A::AbstractArray, ndims_codomain::Val; kwargs...) -> X, Y
    factorize(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> X, Y

Compute the decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - `orth::Symbol=:left`: specify the orthogonality of the decomposition.
    Currently only `:left` and `:right` are supported.
  - Other keywords are passed on directly to MatrixAlgebraKit.
"""
factorize

"""
    eigen(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eigen(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D, V
    eigen(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D, V
    eigen(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - `ishermitian::Bool`: specify if the matrix is Hermitian, which can be used to speed up the
    computation. If `false`, the output `eltype` will always be `<:Complex`.
  - `trunc`: Truncation keywords for `eig(h)_trunc`.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.eig_full!`, `MatrixAlgebraKit.eig_trunc!`, `MatrixAlgebraKit.eig_vals!`,
`MatrixAlgebraKit.eigh_full!`, `MatrixAlgebraKit.eigh_trunc!`, and `MatrixAlgebraKit.eigh_vals!`.
"""
eigen

function eigen!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    # tensor to matrix
    A_mat = matricize(style, A, ndims_codomain)
    D, V = MatrixAlgebra.eigen!!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, = blocks(blockpermute(axes(A), biperm))
    axes_V = tuplemortar((axes_codomain, (axes(V, ndims(V)),)))
    # TODO: Make sure `D` has the same basis as `V`.
    return D, unmatricize(style, V, axes_V)
end
function eigen!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigen!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function eigen(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigen!!(style, copy(A), ndims_codomain; kwargs...)
end
function eigen(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigen!!(copy(A), ndims_codomain; kwargs...)
end

"""
    eigvals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eigvals(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D
    eigvals(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D
    eigvals(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation. The output is a vector of eigenvalues.

## Keyword arguments

  - `ishermitian::Bool`: specify if the matrix is Hermitian, which can be used to speed up the
    computation. If `false`, the output `eltype` will always be `<:Complex`.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.eig_vals!` and `MatrixAlgebraKit.eigh_vals!`.
"""
eigvals

function eigvals!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    return MatrixAlgebra.eigvals!!(A_mat; kwargs...)
end
function eigvals!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigvals!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function eigvals(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigvals!!(style, copy(A), ndims_codomain; kwargs...)
end
function eigvals(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return eigvals!!(copy(A), ndims_codomain; kwargs...)
end

"""
    svd(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> U, S, Vᴴ
    svd(A::AbstractArray, ndims_codomain::Val; kwargs...) -> U, S, Vᴴ
    svd(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> U, S, Vᴴ

Compute the SVD decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - `full::Bool=false`: select between a "thick" or a "thin" decomposition, where both `U` and `Vᴴ`
    are unitary or isometric.
  - `trunc`: Truncation keywords for `svd_trunc`. Not compatible with `full=true`.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.svd_full!`, `MatrixAlgebraKit.svd_compact!`, and `MatrixAlgebraKit.svd_trunc!`.
"""
svd

function svd!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    U, S, Vᴴ = MatrixAlgebra.svd!!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = blocks(blockpermute(axes(A), biperm))
    axes_U = tuplemortar((axes_codomain, (axes(U, 2),)))
    axes_S = tuplemortar(((axes(S, 1),), (axes(S, 2),)))
    axes_Vᴴ = tuplemortar(((axes(Vᴴ, 1),), axes_domain))
    return unmatricize(style, U, axes_U),
        unmatricize(style, S, axes_S),
        unmatricize(style, Vᴴ, axes_Vᴴ)
end
function svd!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return svd!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function svd(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return svd!!(style, copy(A), ndims_codomain; kwargs...)
end
function svd(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return svd!!(copy(A), ndims_codomain; kwargs...)
end

"""
    svdvals(A::AbstractArray, labels_A, labels_codomain, labels_domain) -> S
    svdvals(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}) -> S
    svdvals(A::AbstractArray, ndims_codomain::Val) -> S
    svdvals(A::AbstractArray, biperm::AbstractBlockPermutation{2}) -> S

Compute the singular values of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation. The output is a vector of singular values.

See also `MatrixAlgebraKit.svd_vals!`.
"""
svdvals

function svdvals!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val)
    A_mat = matricize(style, A, ndims_codomain)
    return MatrixAlgebra.svdvals!!(A_mat)
end
function svdvals!!(A::AbstractArray, ndims_codomain::Val)
    return svdvals!!(FusionStyle(A), A, ndims_codomain)
end

function svdvals(style::FusionStyle, A::AbstractArray, ndims_codomain::Val)
    return svdvals!!(style, copy(A), ndims_codomain)
end
function svdvals(A::AbstractArray, ndims_codomain::Val)
    return svdvals!!(copy(A), ndims_codomain)
end

"""
    left_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> N
    left_null(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> N
    left_null(A::AbstractArray, ndims_codomain::Val; kwargs...) -> N
    left_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> N

Compute the left nullspace of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.
The output satisfies `N' * A ≈ 0` and `N' * N ≈ I`.

## Keyword arguments

  - `atol::Real=0`: absolute tolerance for the nullspace computation.
  - `rtol::Real=0`: relative tolerance for the nullspace computation.
  - `kind::Symbol`: specify the kind of decomposition used to compute the nullspace.
    The options are `:qr`, `:qrpos` and `:svd`. The former two require `0 == atol == rtol`.
    The default is `:qrpos` if `atol == rtol == 0`, and `:svd` otherwise.
"""
left_null

function left_null!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    N = MatrixAlgebraKit.left_null!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain = first(blocks(blockpermute(axes(A), biperm)))
    axes_N = tuplemortar((axes_codomain, (axes(N, 2),)))
    return unmatricize(style, N, axes_N)
end
function left_null!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return left_null!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function left_null(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return left_null!!(style, copy(A), ndims_codomain; kwargs...)
end
function left_null(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return left_null!!(copy(A), ndims_codomain; kwargs...)
end

"""
    right_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Nᴴ
    right_null(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Nᴴ
    right_null(A::AbstractArray, ndims_codomain::Val::Val; kwargs...) -> Nᴴ
    right_null(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> Nᴴ

Compute the right nullspace of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.
The output satisfies `A * Nᴴ' ≈ 0` and `Nᴴ * Nᴴ' ≈ I`.

## Keyword arguments

  - `atol::Real=0`: absolute tolerance for the nullspace computation.
  - `rtol::Real=0`: relative tolerance for the nullspace computation.
  - `kind::Symbol`: specify the kind of decomposition used to compute the nullspace.
    The options are `:lq`, `:lqpos` and `:svd`. The former two require `0 == atol == rtol`.
    The default is `:lqpos` if `atol == rtol == 0`, and `:svd` otherwise.
"""
right_null

function right_null!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    Nᴴ = MatrixAlgebraKit.right_null!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_domain = last(blocks((blockpermute(axes(A), biperm))))
    axes_Nᴴ = tuplemortar(((axes(Nᴴ, 1),), axes_domain))
    return unmatricize(style, Nᴴ, axes_Nᴴ)
end
function right_null!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return right_null!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function right_null(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return right_null!!(style, copy(A), ndims_codomain; kwargs...)
end
function right_null(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return right_null!!(copy(A), ndims_codomain; kwargs...)
end

"""
    gram_eigh_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> X
    gram_eigh_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> X
    gram_eigh_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> X
    gram_eigh_full(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> X

Gram factorization of a generic N-dimensional array, interpreting it as a
Hermitian positive semi-definite linear map from the domain to the codomain
dimensions. Returns `X` such that `A ≈ X * X'` (contracted on the rank leg),
i.e. the codomain axes of `X` match the codomain axes of `A` and `X` has a
single trailing rank axis.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(MatrixAlgebra._clamp_kwargs_doc("A"))

# Examples

```jldoctest
julia> using TensorAlgebra: contract, gram_eigh_full

julia> B = randn(3, 2, 2);

julia> A = contract((:a, :b, :c, :d), conj(B), (:r, :a, :b), B, (:r, :c, :d));

julia> X = gram_eigh_full(A, (:a, :b, :c, :d), (:a, :b), (:c, :d));

julia> A ≈ contract((:a, :b, :c, :d), X, (:a, :b, :r), conj(X), (:c, :d, :r))
true
```

See also [`gram_eigh_full_with_pinv`](@ref) and
[`MatrixAlgebra.gram_eigh_full`](@ref).
"""
gram_eigh_full

function gram_eigh_full!!(
        style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    A_mat = matricize(style, A, ndims_codomain)
    X = MatrixAlgebra.gram_eigh_full!!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain = first(blocks(blockpermute(axes(A), biperm)))
    axes_X = tuplemortar((axes_codomain, (axes(X, 2),)))
    return unmatricize(style, X, axes_X)
end
function gram_eigh_full!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return gram_eigh_full!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function gram_eigh_full(
        style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    return gram_eigh_full!!(style, copy(A), ndims_codomain; kwargs...)
end
function gram_eigh_full(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return gram_eigh_full!!(copy(A), ndims_codomain; kwargs...)
end

"""
    gram_eigh_full_with_pinv(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> X, Y
    gram_eigh_full_with_pinv(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> X, Y
    gram_eigh_full_with_pinv(A::AbstractArray, ndims_codomain::Val; kwargs...) -> X, Y
    gram_eigh_full_with_pinv(A::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...) -> X, Y

Like [`gram_eigh_full`](@ref), but additionally returns `Y ≈ pinv(X)` such
that `Y * X ≈ I` on the rank subspace (a left inverse). The codomain axes
of `X` match the codomain axes of `A`; `Y` has a leading rank axis followed
by the codomain axes.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(MatrixAlgebra._clamp_kwargs_doc("A"))

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using TensorAlgebra: contract, gram_eigh_full_with_pinv

julia> B = randn(8, 2, 2);

julia> A = contract((:a, :b, :c, :d), conj(B), (:r, :a, :b), B, (:r, :c, :d));

julia> X, Y = gram_eigh_full_with_pinv(A, (:a, :b, :c, :d), (:a, :b), (:c, :d));

julia> A ≈ contract((:a, :b, :c, :d), X, (:a, :b, :r), conj(X), (:c, :d, :r))
true

julia> contract((:r, :s), Y, (:r, :a, :b), X, (:a, :b, :s)) ≈ I
true
```

See also [`MatrixAlgebra.gram_eigh_full_with_pinv`](@ref).
"""
gram_eigh_full_with_pinv

function gram_eigh_full_with_pinv!!(
        style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    A_mat = matricize(style, A, ndims_codomain)
    X, Y = MatrixAlgebra.gram_eigh_full_with_pinv!!(A_mat; kwargs...)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain = first(blocks(blockpermute(axes(A), biperm)))
    axes_X = tuplemortar((axes_codomain, (axes(X, 2),)))
    axes_Y = tuplemortar(((axes(Y, 1),), conj.(axes_codomain)))
    return unmatricize(style, X, axes_X), unmatricize(style, Y, axes_Y)
end
function gram_eigh_full_with_pinv!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return gram_eigh_full_with_pinv!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function gram_eigh_full_with_pinv(
        style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    return gram_eigh_full_with_pinv!!(style, copy(A), ndims_codomain; kwargs...)
end
function gram_eigh_full_with_pinv(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return gram_eigh_full_with_pinv!!(copy(A), ndims_codomain; kwargs...)
end

"""
    TensorAlgebra.one(A::AbstractArray, labels_A, labels_codomain, labels_domain) -> Id
    TensorAlgebra.one(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}) -> Id
    TensorAlgebra.one(A::AbstractArray, ndims_codomain::Val) -> Id
    TensorAlgebra.one(A::AbstractArray, biperm::AbstractBlockPermutation{2}) -> Id

Construct the identity operator tensor whose shape mirrors `A`, interpreted as a
linear map from the domain to the codomain dimensions. The codomain and domain
partition is specified either via labels or directly through a bi-permutation;
fused codomain and domain sizes must match. `A` is treated as a shape prototype
and is not mutated.

Not exported, since exporting would clash with the implicit `Base.one`. Qualify
as `TensorAlgebra.one(A, ...)`.

See also `MatrixAlgebraKit.one!`.

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using TensorAlgebra: TensorAlgebra, matricize

julia> A = randn(2, 3, 2, 3);

julia> Id = TensorAlgebra.one(A, (:a, :b, :c, :d), (:a, :b), (:c, :d));

julia> matricize(Id, Val(2)) ≈ I
true
```
"""
one

function one!!(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    MatrixAlgebraKit.one!(A_mat)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = blocks(blockpermute(axes(A), biperm))
    return unmatricize(style, A_mat, axes_codomain, axes_domain)
end
function one!!(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return one!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function one(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
    return one!!(style, copy(A), ndims_codomain; kwargs...)
end
function one(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return one!!(copy(A), ndims_codomain; kwargs...)
end
