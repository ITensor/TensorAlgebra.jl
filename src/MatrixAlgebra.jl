module MatrixAlgebra

export gram_eigh_full,
    gram_eigh_full_with_pinv,
    invsqrt_diag_safe,
    invsqrth_safe,
    pow_diag_safe,
    powh_safe,
    sqrt_diag_safe,
    sqrth_safe,
    sqrth_invsqrth_safe

using LinearAlgebra: LinearAlgebra, Diagonal, isdiag, norm
using MatrixAlgebraKit: MatrixAlgebraKit as MAK

function _clamp_kwargs_doc(arg::AbstractString)
    return join(
        (
            "  - `atol::Real`: absolute clamping threshold. Default `0`.",
            "  - `rtol::Real`: relative clamping threshold. Default `eps(real(eltype($arg)))^(3//4)` when `atol = 0`, else `0`.",
        ), "\n"
    )
end

"""
    pow_diag_safe(D, p; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^p

Raise a diagonal-structured matrix `D` to the power `p`. Diagonal entries
`d` with `abs(d) < tol` are clamped to zero before exponentiation, where
`tol = max(atol, rtol * norm(D, Inf))` (the largest-magnitude entry, which
is the largest-magnitude diagonal entry for a diagonal-structured matrix).
Negative `d` above `tol` cause `d^p` to error for fractional `p` (e.g.
`p = 1//2`) and pass through for integer `p`, so the operation itself
enforces the PSD precondition per-power. Errors if `isdiag(D)` is `false`.

The implementation writes the clamped powers back through `MAK.diagview`
onto a `copy` of `D`, so the result has the input's type and structure, and
types extending `diagview` (e.g. graded or block diagonal, a `TensorMap`)
automatically extend [`sqrt_diag_safe`](@ref), [`invsqrt_diag_safe`](@ref),
and the [`powh_safe`](@ref) family.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
function pow_diag_safe(
        D, p;
        atol = zero(real(eltype(D))),
        rtol = iszero(atol) ? eps(real(eltype(D)))^(3 // 4) :
            zero(real(eltype(D)))
    )
    isdiag(D) || throw(
        ArgumentError("pow_diag_safe requires a diagonal-structured matrix")
    )
    tol = max(atol, rtol * norm(D, Inf))
    Dp = copy(D)
    _pow_diag!(MAK.diagview(Dp), p, tol)
    return Dp
end

# `copyto!` rather than `.=`: block-structured diagonal views (e.g. a graded fused
# vector) have a blockwise `map` and `copyto!` but no broadcast support.
function _pow_diag!(σ, p, tol)
    copyto!(σ, map(d -> abs(d) < tol ? zero(d) : real(d)^p, σ))
    return σ
end
# A backend's `diagview` may be a dict of per-block diagonal views (e.g. a `TensorMap`,
# keyed by sector) rather than a single vector view.
function _pow_diag!(σ::AbstractDict, p, tol)
    foreach(v -> _pow_diag!(v, p, tol), values(σ))
    return σ
end

"""
    sqrt_diag_safe(D; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^(1//2)

Square root of a diagonal-structured matrix `D`, equivalent to
`pow_diag_safe(D, 1//2; atol, rtol)`.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
sqrt_diag_safe(D; kwargs...) = pow_diag_safe(D, 1 // 2; kwargs...)

"""
    invsqrt_diag_safe(D; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^(-1//2)

Inverse square root of a diagonal-structured matrix `D`, treating diagonal
entries below tolerance as zero (Moore-Penrose convention). Equivalent to
`pow_diag_safe(D, -1//2; atol, rtol)`.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
invsqrt_diag_safe(D; kwargs...) = pow_diag_safe(D, -1 // 2; kwargs...)

"""
    powh_safe(M, p; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^p

Raise a Hermitian positive semi-definite matrix to the power `p`. For
diagonal-structured `M` (`isdiag(M) == true`), dispatches to
[`pow_diag_safe`](@ref) and skips the eigendecomposition. Otherwise
computes via `M = V * D * V'` as `V * pow_diag_safe(D, p; atol, rtol) * V'`.

The input must be Hermitian (as for `MatrixAlgebraKit.eigh_full`): project
with `MatrixAlgebraKit.project_hermitian` first if it is Hermitian only up
to numerical noise.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("M"))
"""
function powh_safe(M, p; alg = nothing, kwargs...)
    isdiag(M) && return pow_diag_safe(M, p; kwargs...)
    D, V = MAK.eigh_full(M; alg)
    return V * pow_diag_safe(D, p; kwargs...) * V'
end

"""
    sqrth_safe(M; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^(1//2)

Square root of a Hermitian positive semi-definite matrix.
Equivalent to `powh_safe(M, 1//2; alg, atol, rtol)`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("M"))
"""
sqrth_safe(M; kwargs...) = powh_safe(M, 1 // 2; kwargs...)

"""
    invsqrth_safe(M; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^(-1//2)

Inverse square root of a Hermitian positive semi-definite matrix.
Equivalent to `powh_safe(M, -1//2; alg, atol, rtol)`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("M"))
"""
invsqrth_safe(M; kwargs...) = powh_safe(M, -1 // 2; kwargs...)

"""
    sqrth_invsqrth_safe(M; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^(1//2), M^(-1//2)

Square root and pseudo-inverse square root of a Hermitian positive
semi-definite matrix, from a single eigendecomposition. Equivalent
to `(sqrth_safe(M; ...), invsqrth_safe(M; ...))` but with the
eigendecomposition computed once. Eigenvalues below tolerance are clamped
to zero in both factors (Moore-Penrose convention for the inverse).

The input must be Hermitian (as for `MatrixAlgebraKit.eigh_full`): project
with `MatrixAlgebraKit.project_hermitian` first if it is Hermitian only up
to numerical noise.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("M"))
"""
function sqrth_invsqrth_safe(M; alg = nothing, kwargs...)
    if isdiag(M)
        return pow_diag_safe(M, 1 // 2; kwargs...), pow_diag_safe(M, -1 // 2; kwargs...)
    end
    D, V = MAK.eigh_full(M; alg)
    return V * pow_diag_safe(D, 1 // 2; kwargs...) * V',
        V * pow_diag_safe(D, -1 // 2; kwargs...) * V'
end

for (gram, gram_with_pinv, eigh_full) in (
        (:gram_eigh_full, :gram_eigh_full_with_pinv, :eigh_full),
        (:gram_eigh_full!!, :gram_eigh_full_with_pinv!!, :eigh_full!),
    )
    @eval begin
        function $gram(A::AbstractMatrix; alg = nothing, kwargs...)
            D, V = MAK.$eigh_full(A; alg)
            return V * sqrth_safe(D; kwargs...)
        end
        function $gram_with_pinv(A::AbstractMatrix; alg = nothing, kwargs...)
            D, V = MAK.$eigh_full(A; alg)
            return V * sqrth_safe(D; kwargs...), invsqrth_safe(D; kwargs...) * V'
        end
    end
end

"""
    gram_eigh_full(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X

Gram factorization of a Hermitian positive semi-definite matrix via its
eigendecomposition (balanced eigh): returns `X = V * sqrth_safe(D; atol, rtol)`
such that `A ≈ X * X'`, where `A = V * D * V'`. The square-root of `D` is
absorbed symmetrically into the two factors of the eigendecomposition.
Eigenvalues below `tol` (see [`pow_diag_safe`](@ref)) are clamped to zero.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("A"))

# Examples

```jldoctest
julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full

julia> B = [1.0 0.5; 0.5 2.0];

julia> A = B' * B;

julia> X = gram_eigh_full(A);

julia> X * X' ≈ A
true
```

See also [`gram_eigh_full_with_pinv`](@ref).
"""
gram_eigh_full

"""
    gram_eigh_full_with_pinv(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X, Y

Like [`gram_eigh_full`](@ref), but additionally returns
`Y = invsqrth_safe(D; atol, rtol) * V' ≈ pinv(X)`, a left inverse of `X`
on the rank subspace: `Y * X ≈ I`. Eigenvalues below `tol` are clamped to
zero in both factors.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("A"))

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full_with_pinv

julia> B = [1.0 0.5; 0.5 2.0];

julia> A = B' * B;

julia> X, Y = gram_eigh_full_with_pinv(A);

julia> X * X' ≈ A
true

julia> Y * X ≈ I
true
```
"""
gram_eigh_full_with_pinv

using MatrixAlgebraKit: MatrixAlgebraKit, TruncationStrategy

struct TruncationDegenerate{Strategy <: TruncationStrategy, T <: Real} <: TruncationStrategy
    strategy::Strategy
    atol::T
    rtol::T
end

"""
    truncdegen(trunc::TruncationStrategy; atol::Real=0, rtol::Real=0)

Modify a truncation strategy so that if the truncation falls within
a degenerate subspace, the entire subspace gets truncated as well.
A value `val` is considered degenerate if
`norm(val - truncval) ≤ max(atol, rtol * norm(truncval))`
where `truncval` is the largest value truncated by the original
truncation strategy `trunc`.

For now, this truncation strategy assumes the spectrum being truncated
has already been reverse sorted and the strategy being wrapped
outputs a contiguous subset of values including the largest one. It
also only truncates for now, so may not respect if a minimum dimension
was requested in the strategy being wrapped. These restrictions may
be lifted in the future or provided through a different truncation strategy.
"""
function truncdegen(strategy::TruncationStrategy; atol::Real = 0, rtol::Real = 0)
    return TruncationDegenerate(strategy, promote(atol, rtol)...)
end

using MatrixAlgebraKit: findtruncated
function MatrixAlgebraKit.findtruncated(
        values::AbstractVector, strategy::TruncationDegenerate
    )
    Base.require_one_based_indexing(values)
    issorted(values; rev = true) || throw(ArgumentError("Values must be reverse sorted."))
    indices_collection = findtruncated(values, strategy.strategy)
    indices = Base.OneTo(maximum(indices_collection))
    indices_collection == indices ||
        throw(ArgumentError("Truncation must be a contiguous range."))
    if length(indices_collection) == length(values)
        # No truncation occurred.
        return indices
    end
    # The largest truncated value.
    truncval = values[last(indices) + 1]
    # Tolerance of determining if a value is degenerate.
    atol = max(strategy.atol, strategy.rtol * abs(truncval))
    for rank in reverse(indices)
        ≈(values[rank], truncval; atol, rtol = 0) || return Base.OneTo(rank)
    end
    return Base.OneTo(0)
end

end
