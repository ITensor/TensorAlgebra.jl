module MatrixAlgebra

export eigen,
    eigen!!,
    eigvals,
    eigvals!!,
    factorize,
    factorize!!,
    gram_eigh_full,
    gram_eigh_full!!,
    gram_eigh_full_with_pinv,
    gram_eigh_full_with_pinv!!,
    invsqrt_diag_safe,
    invsqrth_safe,
    lq,
    lq!!,
    orth,
    orth!!,
    polar,
    polar!!,
    pow_diag_safe,
    powh_safe,
    qr,
    qr!!,
    sqrt_diag_safe,
    sqrth_safe,
    svd,
    svd!!,
    svdvals,
    svdvals!!

import MatrixAlgebraKit as MAK
using LinearAlgebra: LinearAlgebra, Diagonal, norm

for (f, f_full, f_compact) in (
        (:qr, :qr_full, :qr_compact),
        (:qr!!, :qr_full!, :qr_compact!),
        (:lq, :lq_full, :lq_compact),
        (:lq!!, :lq_full!, :lq_compact!),
    )
    @eval begin
        function $f(A::AbstractMatrix; full::Bool = false, kwargs...)
            return full ? MAK.$f_full(A; kwargs...) : MAK.$f_compact(A; kwargs...)
        end
    end
end

for (eigen, eigh_full, eig_full, eigh_trunc, eig_trunc) in (
        (:eigen, :eigh_full, :eig_full, :eigh_trunc, :eig_trunc),
        (:eigen!!, :eigh_full!, :eig_full!, :eigh_trunc!, :eig_trunc!),
    )
    @eval begin
        function $eigen(
                A::AbstractMatrix;
                trunc = nothing,
                ishermitian = nothing,
                kwargs...
            )
            ishermitian = @something ishermitian LinearAlgebra.ishermitian(A)
            return if !isnothing(trunc)
                if ishermitian
                    MAK.$eigh_trunc(A; trunc, kwargs...)
                else
                    MAK.$eig_trunc(A; trunc, kwargs...)
                end
            else
                if ishermitian
                    MAK.$eigh_full(A; kwargs...)
                else
                    MAK.$eig_full(A; kwargs...)
                end
            end
        end
    end
end

for (eigvals, eigh_vals, eig_vals) in
    ((:eigvals, :eigh_vals, :eig_vals), (:eigvals!!, :eigh_vals!, :eig_vals!))
    @eval begin
        function $eigvals(A::AbstractMatrix; ishermitian = nothing, kwargs...)
            ishermitian = @something ishermitian LinearAlgebra.ishermitian(A)
            return ishermitian ? MAK.$eigh_vals(A; kwargs...) : MAK.$eig_vals(A; kwargs...)
        end
    end
end

"""
    _clamp_kwargs_doc(arg::AbstractString)

Shared documentation for the `atol` and `rtol` keyword arguments of the
`pow_diag_safe` / `powh_safe` family. `arg` is the name of the matrix
argument used in the signatures of the host docstring, so the default
`rtol` formula reads against the right variable.
"""
function _clamp_kwargs_doc(arg::AbstractString)
    return join(
        (
            "  - `atol::Real`: absolute clamping threshold. Default `0`.",
            "  - `rtol::Real`: relative clamping threshold. Default `eps(real(eltype($arg)))^(3//4)` when `atol = 0`, else `0`.",
        ), "\n"
    )
end

"""
    pow_diag_safe(D::Diagonal, p; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^p

Raise a diagonal matrix `D` to the power `p`. Diagonal entries `d` with
`abs(d) < tol` are clamped to zero before exponentiation, where
`tol = max(atol, rtol * maximum(abs, D.diag))`. Negative `d` above `tol`
cause `d^p` to error for fractional `p` (e.g. `p = 1//2`) and pass
through for integer `p`, so the operation itself enforces the PSD
precondition per-power.

This is the leaf operation for diagonal-like types: extending it to a
new diagonal-like type (e.g. graded or block diagonal) automatically
extends [`sqrt_diag_safe`](@ref), [`invsqrt_diag_safe`](@ref), and the
[`powh_safe`](@ref) family.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
function pow_diag_safe(
        D::Diagonal, p;
        atol = zero(real(eltype(D))),
        rtol = iszero(atol) ? eps(real(eltype(D)))^(3 // 4) :
            zero(real(eltype(D)))
    )
    σ = D.diag
    tol = max(atol, rtol * maximum(abs, σ; init = zero(real(eltype(D)))))
    return Diagonal(map(d -> abs(d) < tol ? zero(d) : real(d)^p, σ))
end

"""
    sqrt_diag_safe(D; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^(1//2)

Square root of a diagonal matrix `D`, equivalent to
`pow_diag_safe(D, 1//2; atol, rtol)`.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
sqrt_diag_safe(D; kwargs...) = pow_diag_safe(D, 1 // 2; kwargs...)

"""
    invsqrt_diag_safe(D; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^(-1//2)

Inverse square root of a diagonal matrix `D`, treating diagonal entries
below tolerance as zero (Moore-Penrose convention). Equivalent to
`pow_diag_safe(D, -1//2; atol, rtol)`.

## Keyword arguments

$(_clamp_kwargs_doc("D"))
"""
invsqrt_diag_safe(D; kwargs...) = pow_diag_safe(D, -1 // 2; kwargs...)

"""
    powh_safe(M::AbstractMatrix, p; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^p
    powh_safe(D::Diagonal, p; atol=0, rtol=eps(real(eltype(D)))^(3//4)) -> D^p

Raise an approximately Hermitian positive semi-definite matrix to the
power `p`. For a general `M`, this is computed via the eigendecomposition
`M = V * D * V'` as `V * pow_diag_safe(D, p; atol, rtol) * V'`. For a
`Diagonal` input, this dispatches to [`pow_diag_safe`](@ref).

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full` (only used when
    `M` is non-diagonal).

$(_clamp_kwargs_doc("M"))
"""
powh_safe(D::Diagonal, p; kwargs...) = pow_diag_safe(D, p; kwargs...)

function powh_safe(M::AbstractMatrix, p; alg = nothing, kwargs...)
    D, V = MAK.eigh_full(M, MAK.select_algorithm(MAK.eigh_full, M, alg))
    return V * pow_diag_safe(D, p; kwargs...) * V'
end

"""
    sqrth_safe(M; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^(1//2)

Square root of an approximately Hermitian positive semi-definite matrix.
Equivalent to `powh_safe(M, 1//2; alg, atol, rtol)`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full` (only used when
    `M` is non-diagonal).

$(_clamp_kwargs_doc("M"))
"""
sqrth_safe(M; kwargs...) = powh_safe(M, 1 // 2; kwargs...)

"""
    invsqrth_safe(M; alg=nothing, atol=0, rtol=eps(real(eltype(M)))^(3//4)) -> M^(-1//2)

Inverse square root of an approximately Hermitian positive semi-definite
matrix. Equivalent to `powh_safe(M, -1//2; alg, atol, rtol)`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full` (only used when
    `M` is non-diagonal).

$(_clamp_kwargs_doc("M"))
"""
invsqrth_safe(M; kwargs...) = powh_safe(M, -1 // 2; kwargs...)

for (gram, gram_with_pinv, eigh_full) in (
        (:gram_eigh_full, :gram_eigh_full_with_pinv, :eigh_full),
        (:gram_eigh_full!!, :gram_eigh_full_with_pinv!!, :eigh_full!),
    )
    @eval begin
        function $gram(A::AbstractMatrix; alg = nothing, kwargs...)
            D, V = MAK.$eigh_full(A, MAK.select_algorithm(MAK.$eigh_full, A, alg))
            return sqrth_safe(D; kwargs...) * V'
        end
        function $gram_with_pinv(A::AbstractMatrix; alg = nothing, kwargs...)
            D, V = MAK.$eigh_full(A, MAK.select_algorithm(MAK.$eigh_full, A, alg))
            return sqrth_safe(D; kwargs...) * V', V * invsqrth_safe(D; kwargs...)
        end
    end
end

"""
    gram_eigh_full(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X
    gram_eigh_full!!(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X

Gram factorization of a Hermitian positive semi-definite matrix via its
eigendecomposition: returns `X = sqrth_safe(D; atol, rtol) * V'` such
that `A ≈ X' * X`, where `A = V * D * V'`. Eigenvalues below `tol` (see
[`pow_diag_safe`](@ref)) are clamped to zero. The `!!` variant may
destroy `A`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("A"))

See also [`gram_eigh_full_with_pinv`](@ref).
"""
gram_eigh_full, gram_eigh_full!!

"""
    gram_eigh_full_with_pinv(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X, Y
    gram_eigh_full_with_pinv!!(A::AbstractMatrix; alg=nothing, atol=0, rtol=eps(real(eltype(A)))^(3//4)) -> X, Y

Like [`gram_eigh_full`](@ref), but additionally returns
`Y = V * invsqrth_safe(D; atol, rtol) ≈ pinv(X)` so that `X * Y ≈ I` on
the rank subspace. Eigenvalues below `tol` are clamped to zero in both
factors. The `!!` variant may destroy `A`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.

$(_clamp_kwargs_doc("A"))
"""
gram_eigh_full_with_pinv, gram_eigh_full_with_pinv!!

for (svd, svd_trunc, svd_full, svd_compact) in (
        (:svd, :svd_trunc, :svd_full, :svd_compact),
        (:svd!!, :svd_trunc!, :svd_full!, :svd_compact!),
    )
    _svd = Symbol(:_, svd)
    @eval begin
        function $svd(
                A::AbstractMatrix; full::Union{Bool, Val} = Val(false), trunc = nothing,
                kwargs...
            )
            return $_svd(full, trunc, A; kwargs...)
        end
        function $_svd(full::Bool, trunc, A::AbstractMatrix; kwargs...)
            return $_svd(Val(full), trunc, A; kwargs...)
        end
        function $_svd(full::Val{false}, trunc::Nothing, A::AbstractMatrix; kwargs...)
            return MAK.$svd_compact(A; kwargs...)
        end
        function $_svd(full::Val{false}, trunc, A::AbstractMatrix; kwargs...)
            return MAK.$svd_trunc(A; trunc, kwargs...)
        end
        function $_svd(full::Val{true}, trunc::Nothing, A::AbstractMatrix; kwargs...)
            return MAK.$svd_full(A; kwargs...)
        end
        function $_svd(full::Val{true}, trunc, A::AbstractMatrix; kwargs...)
            return throw(
                ArgumentError(
                    "Specified both full and truncation, currently not supported"
                )
            )
        end
    end
end

for (svdvals, svd_vals) in ((:svdvals, :svd_vals), (:svdvals!!, :svd_vals!))
    @eval begin
        function $svdvals(A::AbstractMatrix; ishermitian = nothing, kwargs...)
            return MAK.$svd_vals(A; kwargs...)
        end
    end
end

for (polar, left_polar, right_polar) in
    ((:polar, :left_polar, :right_polar), (:polar!!, :left_polar!, :right_polar!))
    @eval begin
        function $polar(A::AbstractMatrix; side = :left, kwargs...)
            return if side == :left
                MAK.$left_polar(A; kwargs...)
            elseif side == :right
                MAK.$right_polar(A; kwargs...)
            else
                throw(ArgumentError("`side = $side` not supported."))
            end
        end
    end
end

for (orth, left_orth, right_orth) in
    ((:orth, :left_orth, :right_orth), (:orth!!, :left_orth!, :right_orth!))
    @eval begin
        function $orth(A::AbstractMatrix; side = :left, kwargs...)
            return if side == :left
                MAK.$left_orth(A; kwargs...)
            elseif side == :right
                MAK.$right_orth(A; kwargs...)
            else
                throw(ArgumentError("`side = $side` not supported."))
            end
        end
    end
end

for (factorize, orth_f) in ((:factorize, :(MatrixAlgebra.orth)), (:factorize!!, :orth!!))
    @eval begin
        function $factorize(A::AbstractMatrix; orth = :left, kwargs...)
            return if orth in (:left, :right)
                $orth_f(A; side = orth, kwargs...)
            else
                throw(ArgumentError("`orth = $orth` not supported."))
            end
        end
    end
end

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
