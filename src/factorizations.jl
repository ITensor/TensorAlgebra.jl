using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit

# Two-output factorizations: the first factor `X` has the codomain axes plus a trailing
# rank axis, the second factor `Y` has a leading rank axis plus the domain axes.
for f in (
        :qr_compact, :qr_full, :lq_compact, :lq_full,
        :left_polar, :right_polar, :left_orth, :right_orth,
    )
    @eval begin
        function $f(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            X, Y = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain, axes_domain = bipartition(axes(A), ndims_codomain)
            return unmatricize(style, X, axes_codomain, (axes(X, 2),)),
                unmatricize(style, Y, (axes(Y, 1),), axes_domain)
        end
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

for f in (
        :qr_compact, :qr_full, :lq_compact, :lq_full,
        :left_polar, :right_polar, :left_orth, :right_orth,
        :svd_compact, :svd_full, :svd_trunc, :svd_vals,
        :eigh_full, :eig_full, :eigh_trunc, :eig_trunc, :eigh_vals, :eig_vals,
        :left_null, :right_null, :gram_eigh_full, :gram_eigh_full_with_pinv, :one,
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
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(style, A, perm_codomain, perm_domain; kwargs...)
        end
        function $f(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(A, perm_codomain, perm_domain; kwargs...)
        end
    end
end

"""
    qr_compact(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr_compact(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Q, R
    qr_compact(A::AbstractArray, ndims_codomain::Val; kwargs...) -> Q, R

Compute the compact QR decomposition of a generic N-dimensional array, by interpreting it
as a linear map from the domain to the codomain dimensions, where `R` is square. The
partition is specified either via labels or directly through a bi-permutation.

## Keyword arguments

  - `positive::Bool=false`: specify if the diagonal of `R` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.qr_compact!`.
"""
qr_compact

"""
    qr_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Q, R
    qr_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> Q, R

Compute the full QR decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions, where `Q` is unitary. The
partition is specified either via labels or directly through a bi-permutation.

## Keyword arguments

  - `positive::Bool=false`: specify if the diagonal of `R` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.qr_full!`.
"""
qr_full

"""
    lq_compact(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq_compact(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> L, Q
    lq_compact(A::AbstractArray, ndims_codomain::Val; kwargs...) -> L, Q

Compute the compact LQ decomposition of a generic N-dimensional array, by interpreting it
as a linear map from the domain to the codomain dimensions, where `L` is square. The
partition is specified either via labels or directly through a bi-permutation.

## Keyword arguments

  - `positive::Bool=false`: specify if the diagonal of `L` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.lq_compact!`.
"""
lq_compact

"""
    lq_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> L, Q
    lq_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> L, Q

Compute the full LQ decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions, where `Q` is unitary. The
partition is specified either via labels or directly through a bi-permutation.

## Keyword arguments

  - `positive::Bool=false`: specify if the diagonal of `L` should be positive, leading to a unique decomposition.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.lq_full!`.
"""
lq_full

"""
    left_polar(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> W, P
    left_polar(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> W, P
    left_polar(A::AbstractArray, ndims_codomain::Val; kwargs...) -> W, P

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

Compute the right orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_orth!`.
"""
right_orth

# Three-output SVD: `U` carries the codomain axes plus a trailing rank axis, `S` is the
# rank × rank spectrum, and `Vᴴ` carries a leading rank axis plus the domain axes.
for f in (:svd_compact, :svd_full, :svd_trunc)
    @eval begin
        function $f(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            U, S, Vᴴ = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain, axes_domain = bipartition(axes(A), ndims_codomain)
            return unmatricize(style, U, axes_codomain, (axes(U, 2),)),
                unmatricize(style, S, (axes(S, 1),), (axes(S, 2),)),
                unmatricize(style, Vᴴ, (axes(Vᴴ, 1),), axes_domain)
        end
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

# Eigendecomposition: `D` is the rank × rank spectrum, left as a matrix, while `V` carries
# the codomain axes plus a trailing rank axis.
for f in (:eigh_full, :eig_full, :eigh_trunc, :eig_trunc)
    @eval begin
        function $f(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            D, V = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain = first(bipartition(axes(A), ndims_codomain))
            return D, unmatricize(style, V, axes_codomain, (axes(V, ndims(V)),))
        end
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

# Spectrum-only factorizations returning a vector of singular values / eigenvalues.
for f in (:svd_vals, :eigh_vals, :eig_vals)
    @eval begin
        function $f(style::FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            return MatrixAlgebraKit.$f(A_mat; kwargs...)
        end
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

"""
    svd_compact(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd_compact(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> U, S, Vᴴ
    svd_compact(A::AbstractArray, ndims_codomain::Val; kwargs...) -> U, S, Vᴴ

Compute the compact (thin) SVD of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions, where `U` and `Vᴴ` are isometric.
The partition is specified either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.svd_compact!`.
"""
svd_compact

"""
    svd_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> U, S, Vᴴ
    svd_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> U, S, Vᴴ

Compute the full (thick) SVD of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions, where `U` and `Vᴴ` are unitary.
The partition is specified either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.svd_full!`.
"""
svd_full

"""
    svd_trunc(A::AbstractArray, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> U, S, Vᴴ
    svd_trunc(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> U, S, Vᴴ
    svd_trunc(A::AbstractArray, ndims_codomain::Val; trunc, kwargs...) -> U, S, Vᴴ

Compute the truncated SVD of a generic N-dimensional array, by interpreting it as a linear
map from the domain to the codomain dimensions. The partition is specified either via
labels or directly through a bi-permutation.

## Keyword arguments

  - `trunc`: truncation strategy, passed on to `MatrixAlgebraKit.svd_trunc`.
  - Other keywords are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.svd_trunc!`.
"""
svd_trunc

"""
    svd_vals(A::AbstractArray, labels_A, labels_codomain, labels_domain) -> S
    svd_vals(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}) -> S
    svd_vals(A::AbstractArray, ndims_codomain::Val) -> S

Compute the singular values of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions. The partition is specified either
via labels or directly through a bi-permutation. The output is a vector of singular values.

See also `MatrixAlgebraKit.svd_vals!`.
"""
svd_vals

"""
    eigh_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eigh_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D, V
    eigh_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array interpreted as a
Hermitian linear map from the domain to the codomain dimensions. The partition is specified
either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.eigh_full!`.
"""
eigh_full

"""
    eig_full(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eig_full(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D, V
    eig_full(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array interpreted as a
general (non-Hermitian) linear map from the domain to the codomain dimensions. The output
`eltype` is always `<:Complex`. The partition is specified either via labels or directly
through a bi-permutation.

See also `MatrixAlgebraKit.eig_full!`.
"""
eig_full

"""
    eigh_trunc(A::AbstractArray, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> D, V
    eigh_trunc(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> D, V
    eigh_trunc(A::AbstractArray, ndims_codomain::Val; trunc, kwargs...) -> D, V

Truncated Hermitian eigenvalue decomposition, like [`eigh_full`](@ref) but keeping only the
eigenvalues selected by the `trunc` strategy.

See also `MatrixAlgebraKit.eigh_trunc!`.
"""
eigh_trunc

"""
    eig_trunc(A::AbstractArray, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> D, V
    eig_trunc(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> D, V
    eig_trunc(A::AbstractArray, ndims_codomain::Val; trunc, kwargs...) -> D, V

Truncated general eigenvalue decomposition, like [`eig_full`](@ref) but keeping only the
eigenvalues selected by the `trunc` strategy.

See also `MatrixAlgebraKit.eig_trunc!`.
"""
eig_trunc

"""
    eigh_vals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eigh_vals(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D
    eigh_vals(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array interpreted as a Hermitian linear
map from the domain to the codomain dimensions. The output is a vector of eigenvalues.

See also `MatrixAlgebraKit.eigh_vals!`.
"""
eigh_vals

"""
    eig_vals(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eig_vals(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D
    eig_vals(A::AbstractArray, ndims_codomain::Val; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array interpreted as a general
(non-Hermitian) linear map from the domain to the codomain dimensions. The output is a
vector of eigenvalues with `<:Complex` `eltype`.

See also `MatrixAlgebraKit.eig_vals!`.
"""
eig_vals

"""
    left_null(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...) -> N
    left_null(A::AbstractArray, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> N
    left_null(A::AbstractArray, ndims_codomain::Val; kwargs...) -> N

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
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, N, axes_codomain, (axes(N, 2),))
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
    axes_domain = last(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, Nᴴ, (axes(Nᴴ, 1),), axes_domain)
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
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, X, axes_codomain, (axes(X, 2),))
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
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, X, axes_codomain, (axes(X, 2),)),
        unmatricize(style, Y, (axes(Y, 1),), conj.(axes_codomain))
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
    return unmatricize(style, A_mat, bipartition(axes(A), ndims_codomain)...)
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
