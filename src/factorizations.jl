using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit

# Each factorization reconstructs its factors with `unmatricize`, reading the freshly created
# bond axis off the factor itself: it is the factor's last axis on a codomain factor
# (`[group…, bond]`) and its first axis on a domain factor (`[bond, group…]`), on every backend
# (a fusing backend returns a rank-2 factor, a `TensorMap` keeps the group's original legs). The
# bond is dualized to codomain-facing form (`conj`, a no-op on a dense axis) when it lands on the
# domain side of the reconstruction, matching the `unmatricize`/`similar_map` axis convention.

# Two-output factorizations: the first factor `X` has the codomain axes plus a trailing
# rank axis, the second factor `Y` has a leading rank axis plus the domain axes.
for f in (
        :qr_compact, :qr_full, :lq_compact, :lq_full,
        :left_polar, :right_polar, :left_orth, :right_orth,
    )
    @eval begin
        function $f(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            X, Y = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
            return unmatricize(style, X, axes_codomain, (conj(axes(X, ndims(X))),)),
                unmatricize(style, Y, (axes(Y, 1),), axes_domain)
        end
        function $f(A, ndims_codomain::Val; kwargs...)
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
        :sqrth_safe, :invsqrth_safe, :sqrth_invsqrth_safe, :project_hermitian,
    )
    @eval begin
        function $f(
                style::FusionStyle, A,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            A_perm = bipermutedims(A, perm_codomain, perm_domain)
            return $f(style, A_perm, Val(length(perm_codomain)); kwargs...)
        end
        function $f(
                A,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            A_perm = bipermutedims(A, perm_codomain, perm_domain)
            return $f(A_perm, Val(length(perm_codomain)); kwargs...)
        end

        function $f(
                style::FusionStyle, A,
                labels_A, labels_codomain, labels_domain; kwargs...
            )
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(style, A, perm_codomain, perm_domain; kwargs...)
        end
        function $f(A, labels_A, labels_codomain, labels_domain; kwargs...)
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_A, labels_codomain, labels_domain))...)
            return $f(A, perm_codomain, perm_domain; kwargs...)
        end
    end
end

"""
    TensorAlgebra.tr(A, labels_A, labels_codomain, labels_domain)
    TensorAlgebra.tr(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}})
    TensorAlgebra.tr(A, ndims_codomain::Val)

Trace of a generic N-dimensional array `A` interpreted as a linear map from its domain to its
codomain dimensions. The map is matricized into its square matrix, then the matrix trace is
taken, so the backend's own matrix `tr` (dense, graded, or `TensorMap`) does the work. The
partition is specified via labels, a bi-permutation, or directly as the codomain rank, matching
the factorization entry points.

This is `TensorAlgebra`'s own function, distinct from `LinearAlgebra.tr`; the two-argument and
higher forms take a codomain/domain partition rather than a bare matrix.

# Examples

```jldoctest
julia> using TensorAlgebra: tr

julia> tr([1.0 2.0; 3.0 4.0], Val(1))
5.0
```
"""
function tr(style::FusionStyle, A, ndims_codomain::Val)
    return LinearAlgebra.tr(matricize(style, A, ndims_codomain))
end
function tr(A, ndims_codomain::Val)
    return tr(FusionStyle(A), A, ndims_codomain)
end
function tr(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}})
    A_perm = bipermutedims(A, perm_codomain, perm_domain)
    return tr(A_perm, Val(length(perm_codomain)))
end
function tr(A, labels_A, labels_codomain, labels_domain)
    perm_codomain, perm_domain =
        biperm(Tuple.((labels_A, labels_codomain, labels_domain))...)
    return tr(A, perm_codomain, perm_domain)
end

"""
    qr_compact(A, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr_compact(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Q, R
    qr_compact(A, ndims_codomain::Val; kwargs...) -> Q, R

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
    qr_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> Q, R
    qr_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Q, R
    qr_full(A, ndims_codomain::Val; kwargs...) -> Q, R

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
    lq_compact(A, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq_compact(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> L, Q
    lq_compact(A, ndims_codomain::Val; kwargs...) -> L, Q

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
    lq_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> L, Q
    lq_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> L, Q
    lq_full(A, ndims_codomain::Val; kwargs...) -> L, Q

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
    left_polar(A, labels_A, labels_codomain, labels_domain; kwargs...) -> W, P
    left_polar(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> W, P
    left_polar(A, ndims_codomain::Val; kwargs...) -> W, P

Compute the left polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_polar!`.
"""
left_polar

"""
    right_polar(A, labels_A, labels_codomain, labels_domain; kwargs...) -> P, W
    right_polar(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> P, W
    right_polar(A, ndims_codomain::Val; kwargs...) -> P, W

Compute the right polar decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.right_polar!`.
"""
right_polar

"""
    left_orth(A, labels_A, labels_codomain, labels_domain; kwargs...) -> V, C
    left_orth(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> V, C
    left_orth(A, ndims_codomain::Val; kwargs...) -> V, C

Compute the left orthogonal decomposition of a generic N-dimensional array, by interpreting it as
a linear map from the domain to the codomain dimensions. These can be specified either via
their labels or directly through a bi-permutation.

## Keyword arguments

  - Keyword arguments are passed on directly to MatrixAlgebraKit.

See also `MatrixAlgebraKit.left_orth!`.
"""
left_orth

"""
    right_orth(A, labels_A, labels_codomain, labels_domain; kwargs...) -> C, V
    right_orth(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> C, V
    right_orth(A, ndims_codomain::Val; kwargs...) -> C, V

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
for f in (:svd_compact, :svd_full)
    @eval begin
        function $f(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            U, S, Vᴴ = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
            return unmatricize(style, U, axes_codomain, (conj(axes(U, ndims(U))),)),
                unmatricize(style, S, (axes(S, 1),), (conj(axes(S, 2)),)),
                unmatricize(style, Vᴴ, (axes(Vᴴ, 1),), axes_domain)
        end
        function $f(A, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

# `svd_trunc` matches the three-output SVD but additionally surfaces the truncation error
# `ϵ` (the 2-norm of the discarded singular values, computed by MatrixAlgebraKit without
# catastrophic cancellation), so it is spelled out here rather than sharing the loop above.
function svd_trunc(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    U, S, Vᴴ, ϵ = MatrixAlgebraKit.svd_trunc(A_mat; kwargs...)
    axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
    return unmatricize(style, U, axes_codomain, (conj(axes(U, ndims(U))),)),
        unmatricize(style, S, (axes(S, 1),), (conj(axes(S, 2)),)),
        unmatricize(style, Vᴴ, (axes(Vᴴ, 1),), axes_domain),
        ϵ
end
function svd_trunc(A, ndims_codomain::Val; kwargs...)
    return svd_trunc(FusionStyle(A), A, ndims_codomain; kwargs...)
end

# Eigendecomposition: `D` is the rank × rank spectrum, left as a matrix, while `V` carries
# the codomain axes plus a trailing rank axis.
for f in (:eigh_full, :eig_full, :eigh_trunc, :eig_trunc)
    @eval begin
        function $f(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            D, V = MatrixAlgebraKit.$f(A_mat; kwargs...)
            axes_codomain = first(bipartition(axes(A), ndims_codomain))
            return D, unmatricize(style, V, axes_codomain, (conj(axes(V, ndims(V))),))
        end
        function $f(A, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

# Spectrum-only factorizations returning a vector of singular values / eigenvalues.
for f in (:svd_vals, :eigh_vals, :eig_vals)
    @eval begin
        function $f(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            return MatrixAlgebraKit.$f(A_mat; kwargs...)
        end
        function $f(A, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

"""
    svd_compact(A, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd_compact(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> U, S, Vᴴ
    svd_compact(A, ndims_codomain::Val; kwargs...) -> U, S, Vᴴ

Compute the compact (thin) SVD of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions, where `U` and `Vᴴ` are isometric.
The partition is specified either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.svd_compact!`.
"""
svd_compact

"""
    svd_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> U, S, Vᴴ
    svd_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> U, S, Vᴴ
    svd_full(A, ndims_codomain::Val; kwargs...) -> U, S, Vᴴ

Compute the full (thick) SVD of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions, where `U` and `Vᴴ` are unitary.
The partition is specified either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.svd_full!`.
"""
svd_full

"""
    svd_trunc(A, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> U, S, Vᴴ, ϵ
    svd_trunc(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> U, S, Vᴴ, ϵ
    svd_trunc(A, ndims_codomain::Val; trunc, kwargs...) -> U, S, Vᴴ, ϵ

Compute the truncated SVD of a generic N-dimensional array, by interpreting it as a linear
map from the domain to the codomain dimensions. The partition is specified either via
labels or directly through a bi-permutation. In addition to the factors, returns the
truncation error `ϵ`, the 2-norm of the discarded singular values.

## Keyword arguments

  - `trunc`: truncation strategy, passed on to `MatrixAlgebraKit.svd_trunc`.
  - Other keywords are passed on directly to MatrixAlgebraKit.

# Examples

```jldoctest
julia> using TensorAlgebra: svd_trunc, contract

julia> A = randn(4, 4);

julia> U, S, Vᴴ, ϵ = svd_trunc(A, (:i, :j), (:i,), (:j,));

julia> SV = contract((:u, :j), S, (:u, :v), Vᴴ, (:v, :j));

julia> contract((:i, :j), U, (:i, :u), SV, (:u, :j)) ≈ A
true

julia> isapprox(ϵ, 0; atol = 1e-10)
true
```

See also `MatrixAlgebraKit.svd_trunc!`.
"""
svd_trunc

"""
    svd_vals(A, labels_A, labels_codomain, labels_domain) -> S
    svd_vals(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}) -> S
    svd_vals(A, ndims_codomain::Val) -> S

Compute the singular values of a generic N-dimensional array, by interpreting it as a
linear map from the domain to the codomain dimensions. The partition is specified either
via labels or directly through a bi-permutation. The output is a vector of singular values.

See also `MatrixAlgebraKit.svd_vals!`.
"""
svd_vals

"""
    eigh_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eigh_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D, V
    eigh_full(A, ndims_codomain::Val; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array interpreted as a
Hermitian linear map from the domain to the codomain dimensions. The partition is specified
either via labels or directly through a bi-permutation.

See also `MatrixAlgebraKit.eigh_full!`.
"""
eigh_full

"""
    eig_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> D, V
    eig_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D, V
    eig_full(A, ndims_codomain::Val; kwargs...) -> D, V

Compute the eigenvalue decomposition of a generic N-dimensional array interpreted as a
general (non-Hermitian) linear map from the domain to the codomain dimensions. The output
`eltype` is always `<:Complex`. The partition is specified either via labels or directly
through a bi-permutation.

See also `MatrixAlgebraKit.eig_full!`.
"""
eig_full

"""
    eigh_trunc(A, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> D, V
    eigh_trunc(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> D, V
    eigh_trunc(A, ndims_codomain::Val; trunc, kwargs...) -> D, V

Truncated Hermitian eigenvalue decomposition, like [`eigh_full`](@ref) but keeping only the
eigenvalues selected by the `trunc` strategy.

See also `MatrixAlgebraKit.eigh_trunc!`.
"""
eigh_trunc

"""
    eig_trunc(A, labels_A, labels_codomain, labels_domain; trunc, kwargs...) -> D, V
    eig_trunc(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; trunc, kwargs...) -> D, V
    eig_trunc(A, ndims_codomain::Val; trunc, kwargs...) -> D, V

Truncated general eigenvalue decomposition, like [`eig_full`](@ref) but keeping only the
eigenvalues selected by the `trunc` strategy.

See also `MatrixAlgebraKit.eig_trunc!`.
"""
eig_trunc

"""
    eigh_vals(A, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eigh_vals(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D
    eigh_vals(A, ndims_codomain::Val; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array interpreted as a Hermitian linear
map from the domain to the codomain dimensions. The output is a vector of eigenvalues.

See also `MatrixAlgebraKit.eigh_vals!`.
"""
eigh_vals

"""
    eig_vals(A, labels_A, labels_codomain, labels_domain; kwargs...) -> D
    eig_vals(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> D
    eig_vals(A, ndims_codomain::Val; kwargs...) -> D

Compute the eigenvalues of a generic N-dimensional array interpreted as a general
(non-Hermitian) linear map from the domain to the codomain dimensions. The output is a
vector of eigenvalues with `<:Complex` `eltype`.

See also `MatrixAlgebraKit.eig_vals!`.
"""
eig_vals

"""
    left_null(A, labels_A, labels_codomain, labels_domain; kwargs...) -> N
    left_null(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> N
    left_null(A, ndims_codomain::Val; kwargs...) -> N

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

function left_null!!(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    N = MatrixAlgebraKit.left_null!(A_mat; kwargs...)
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, N, axes_codomain, (conj(axes(N, ndims(N))),))
end
function left_null!!(A, ndims_codomain::Val; kwargs...)
    return left_null!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function left_null(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    return left_null!!(style, copy(A), ndims_codomain; kwargs...)
end
function left_null(A, ndims_codomain::Val; kwargs...)
    return left_null!!(copy(A), ndims_codomain; kwargs...)
end

"""
    right_null(A, labels_A, labels_codomain, labels_domain; kwargs...) -> Nᴴ
    right_null(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> Nᴴ
    right_null(A, ndims_codomain::Val::Val; kwargs...) -> Nᴴ

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

function right_null!!(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    Nᴴ = MatrixAlgebraKit.right_null!(A_mat; kwargs...)
    _, axes_domain = bipartition_axes(axes(A), ndims_codomain)
    return unmatricize(style, Nᴴ, (axes(Nᴴ, 1),), axes_domain)
end
function right_null!!(A, ndims_codomain::Val; kwargs...)
    return right_null!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function right_null(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    return right_null!!(style, copy(A), ndims_codomain; kwargs...)
end
function right_null(A, ndims_codomain::Val; kwargs...)
    return right_null!!(copy(A), ndims_codomain; kwargs...)
end

"""
    gram_eigh_full(A, labels_A, labels_codomain, labels_domain; kwargs...) -> X
    gram_eigh_full(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> X
    gram_eigh_full(A, ndims_codomain::Val; kwargs...) -> X

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
        style::FusionStyle, A, ndims_codomain::Val; kwargs...
    )
    A_mat = matricize(style, A, ndims_codomain)
    X = MatrixAlgebra.gram_eigh_full!!(A_mat; kwargs...)
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, X, axes_codomain, (conj(axes(X, ndims(X))),))
end
function gram_eigh_full!!(A, ndims_codomain::Val; kwargs...)
    return gram_eigh_full!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function gram_eigh_full(
        style::FusionStyle, A, ndims_codomain::Val; kwargs...
    )
    return gram_eigh_full!!(style, copy(A), ndims_codomain; kwargs...)
end
function gram_eigh_full(A, ndims_codomain::Val; kwargs...)
    return gram_eigh_full!!(copy(A), ndims_codomain; kwargs...)
end

"""
    gram_eigh_full_with_pinv(A, labels_A, labels_codomain, labels_domain; kwargs...) -> X, Y
    gram_eigh_full_with_pinv(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> X, Y
    gram_eigh_full_with_pinv(A, ndims_codomain::Val; kwargs...) -> X, Y

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
        style::FusionStyle, A, ndims_codomain::Val; kwargs...
    )
    A_mat = matricize(style, A, ndims_codomain)
    X, Y = MatrixAlgebra.gram_eigh_full_with_pinv!!(A_mat; kwargs...)
    axes_codomain = first(bipartition(axes(A), ndims_codomain))
    return unmatricize(style, X, axes_codomain, (conj(axes(X, ndims(X))),)),
        unmatricize(style, Y, (axes(Y, 1),), axes_codomain)
end
function gram_eigh_full_with_pinv!!(A, ndims_codomain::Val; kwargs...)
    return gram_eigh_full_with_pinv!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function gram_eigh_full_with_pinv(
        style::FusionStyle, A, ndims_codomain::Val; kwargs...
    )
    return gram_eigh_full_with_pinv!!(style, copy(A), ndims_codomain; kwargs...)
end
function gram_eigh_full_with_pinv(A, ndims_codomain::Val; kwargs...)
    return gram_eigh_full_with_pinv!!(copy(A), ndims_codomain; kwargs...)
end

"""
    sqrth_safe(A, labels_A, labels_codomain, labels_domain; kwargs...) -> P
    sqrth_safe(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> P
    sqrth_safe(A, ndims_codomain::Val; kwargs...) -> P

Square root of a generic N-dimensional array, interpreting it as a
Hermitian positive semi-definite linear map from the domain to the
codomain dimensions. The result carries the same codomain and domain axes
as `A`. Eigenvalues below tolerance are clamped to zero. The input must be
Hermitian: project with `project_hermitian` first if it is Hermitian only
up to numerical noise.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(MatrixAlgebra._clamp_kwargs_doc("A"))

See also [`invsqrth_safe`](@ref), [`sqrth_invsqrth_safe`](@ref), and
[`MatrixAlgebra.sqrth_safe`](@ref).
"""
sqrth_safe

"""
    invsqrth_safe(A, labels_A, labels_codomain, labels_domain; kwargs...) -> P
    invsqrth_safe(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> P
    invsqrth_safe(A, ndims_codomain::Val; kwargs...) -> P

Pseudo-inverse square root of a generic N-dimensional array, interpreting
it as a Hermitian positive semi-definite linear map from the domain to the
codomain dimensions. The result carries the same codomain and domain axes
as `A`. Eigenvalues below tolerance are clamped to zero (Moore-Penrose
convention). The input must be Hermitian: project with `project_hermitian`
first if it is Hermitian only up to numerical noise.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(MatrixAlgebra._clamp_kwargs_doc("A"))

See also [`sqrth_safe`](@ref), [`sqrth_invsqrth_safe`](@ref), and
[`MatrixAlgebra.invsqrth_safe`](@ref).
"""
invsqrth_safe

for f in (:sqrth_safe, :invsqrth_safe)
    @eval begin
        function $f(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
            A_mat = matricize(style, A, ndims_codomain)
            P_mat = MatrixAlgebra.$f(A_mat; kwargs...)
            axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
            return unmatricize(style, P_mat, axes_codomain, axes_domain)
        end
        function $f(A, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(A), A, ndims_codomain; kwargs...)
        end
    end
end

"""
    project_hermitian(A, labels_A, labels_codomain, labels_domain; kwargs...) -> H
    project_hermitian(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> H
    project_hermitian(A, ndims_codomain::Val; kwargs...) -> H

Hermitian part `(M + M') / 2` of a generic N-dimensional array, interpreting
it as a linear map `M` from the domain to the codomain dimensions. The result
carries the same codomain and domain axes as `A`.

See also `MatrixAlgebraKit.project_hermitian`.
"""
project_hermitian

function project_hermitian(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    H_mat = MatrixAlgebraKit.project_hermitian(A_mat; kwargs...)
    axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
    return unmatricize(style, H_mat, axes_codomain, axes_domain)
end
function project_hermitian(A, ndims_codomain::Val; kwargs...)
    return project_hermitian(FusionStyle(A), A, ndims_codomain; kwargs...)
end

"""
    sqrth_invsqrth_safe(A, labels_A, labels_codomain, labels_domain; kwargs...) -> P, Pinv
    sqrth_invsqrth_safe(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}; kwargs...) -> P, Pinv
    sqrth_invsqrth_safe(A, ndims_codomain::Val; kwargs...) -> P, Pinv

Square root and pseudo-inverse square root of a generic N-dimensional
array (see [`sqrth_safe`](@ref) and [`invsqrth_safe`](@ref)), from a
single eigendecomposition. Both results carry the same codomain and
domain axes as `A`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(MatrixAlgebra._clamp_kwargs_doc("A"))

See also [`MatrixAlgebra.sqrth_invsqrth_safe`](@ref).
"""
sqrth_invsqrth_safe

function sqrth_invsqrth_safe(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    P_mat, Pinv_mat = MatrixAlgebra.sqrth_invsqrth_safe(A_mat; kwargs...)
    axes_codomain, axes_domain = bipartition_axes(axes(A), ndims_codomain)
    return unmatricize(style, P_mat, axes_codomain, axes_domain),
        unmatricize(style, Pinv_mat, axes_codomain, axes_domain)
end
function sqrth_invsqrth_safe(A, ndims_codomain::Val; kwargs...)
    return sqrth_invsqrth_safe(FusionStyle(A), A, ndims_codomain; kwargs...)
end

"""
    TensorAlgebra.one(A, labels_A, labels_codomain, labels_domain) -> Id
    TensorAlgebra.one(A, perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}) -> Id
    TensorAlgebra.one(A, ndims_codomain::Val) -> Id

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

function one!!(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    MatrixAlgebraKit.one!(A_mat)
    codomain_axes, domain_axes = bipartition_axes(axes(A), ndims_codomain)
    return unmatricize(style, A_mat, codomain_axes, domain_axes)
end
function one!!(A, ndims_codomain::Val; kwargs...)
    return one!!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

# In-place identity fill: writes the identity into `A` and returns it. Matricizes `A`, fills the
# fused matrix with the identity, and — when the matricized form is a detached copy (a graded
# gather) rather than a view aliasing `A` (a dense reshape) — scatters it back with `unmatricize!`.
function one!(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    A_mat = matricize(style, A, ndims_codomain)
    MatrixAlgebraKit.one!(A_mat)
    Base.mightalias(A_mat, A) && return A
    return unmatricize!(A, A_mat, ndims_codomain)
end
function one!(A, ndims_codomain::Val; kwargs...)
    return one!(FusionStyle(A), A, ndims_codomain; kwargs...)
end

function one(style::FusionStyle, A, ndims_codomain::Val; kwargs...)
    return one!!(style, copy(A), ndims_codomain; kwargs...)
end
function one(A, ndims_codomain::Val; kwargs...)
    return one!!(copy(A), ndims_codomain; kwargs...)
end
