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
    lq,
    lq!!,
    orth,
    orth!!,
    polar,
    polar!!,
    qr,
    qr!!,
    svd,
    svd!!,
    svdvals,
    svdvals!!

import MatrixAlgebraKit as MAK
using LinearAlgebra: LinearAlgebra, Diagonal, diag, norm

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
    pinv_tol(λ; atol=0, rtol=...) -> tol
    pinv_tol(λ, pinv::NamedTuple) -> tol

Tolerance used by [`gram_eigh_full`](@ref) and
[`gram_eigh_full_with_pinv`](@ref) to clamp small eigenvalues to zero:
`tol = max(atol, rtol * maximum(abs, λ))`. The `NamedTuple` form splats
its fields as keyword arguments.
"""
pinv_tol(λ, pinv::NamedTuple) = pinv_tol(λ; pinv...)
function pinv_tol(
        λ; atol = zero(real(eltype(λ))),
        rtol = iszero(atol) ? eps(real(eltype(λ))) * length(λ) :
            zero(real(eltype(λ)))
    )
    return max(atol, rtol * maximum(abs, λ; init = zero(real(eltype(λ)))))
end

"""
    sqrt_safe(a::Number, tol=MatrixAlgebraKit.defaulttol(a))

Compute `sqrt(a)` when `abs(a) ≥ tol`, otherwise return `zero(a)`.
"""
sqrt_safe(a::Number, tol = MAK.defaulttol(a)) = abs(a) < tol ? zero(a) : sqrt(a)

for (gram, gram_with_pinv, eigh_full) in (
        (:gram_eigh_full, :gram_eigh_full_with_pinv, :eigh_full),
        (:gram_eigh_full!!, :gram_eigh_full_with_pinv!!, :eigh_full!),
    )
    @eval begin
        function $gram(A::AbstractMatrix; alg = nothing, pinv = (;))
            D, V = MAK.$eigh_full(A, MAK.select_algorithm(MAK.$eigh_full, A, alg))
            λ = diag(D)
            sqrtλ = map(l -> sqrt_safe(l, pinv_tol(λ, pinv)), λ)
            return Diagonal(sqrtλ) * V'
        end
        function $gram_with_pinv(A::AbstractMatrix; alg = nothing, pinv = (;))
            D, V = MAK.$eigh_full(A, MAK.select_algorithm(MAK.$eigh_full, A, alg))
            λ = diag(D)
            sqrtλ = map(l -> sqrt_safe(l, pinv_tol(λ, pinv)), λ)
            inv_sqrtλ = map(s -> iszero(s) ? s : inv(s), sqrtλ)
            return Diagonal(sqrtλ) * V', V * Diagonal(inv_sqrtλ)
        end
    end
end

"""
    gram_eigh_full(A::AbstractMatrix; alg=nothing, pinv=(;)) -> X
    gram_eigh_full!!(A::AbstractMatrix; alg=nothing, pinv=(;)) -> X

Gram factorization of a Hermitian positive semi-definite matrix via its
eigendecomposition: returns `X = Diagonal(sqrt.(Λ)) * V'` such that
`A ≈ X' * X`, where `A = V * Diagonal(Λ) * V'`. Eigenvalues below
[`pinv_tol`](@ref) are clamped to zero. The `!!` variant may destroy `A`.

## Keyword arguments

  - `alg`: forwarded to `MatrixAlgebraKit.eigh_full`.
  - `pinv::NamedTuple`: forwarded to [`pinv_tol`](@ref) (e.g. `(; atol, rtol)`).

See also [`gram_eigh_full_with_pinv`](@ref).
"""
gram_eigh_full, gram_eigh_full!!

"""
    gram_eigh_full_with_pinv(A::AbstractMatrix; alg=nothing, pinv=(;)) -> X, Y
    gram_eigh_full_with_pinv!!(A::AbstractMatrix; alg=nothing, pinv=(;)) -> X, Y

Like [`gram_eigh_full`](@ref), but additionally returns
`Y = V * Diagonal(inv.(sqrt.(Λ))) ≈ pinv(X)` so that `X * Y ≈ I` on the
rank subspace. Eigenvalues below [`pinv_tol`](@ref) are clamped to zero
in both factors. The `!!` variant may destroy `A`.
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
