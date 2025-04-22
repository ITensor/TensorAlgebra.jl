module MatrixAlgebra

export eigen,
  eigen!,
  eigvals,
  eigvals!,
  factorize,
  factorize!,
  lq,
  lq!,
  orth,
  orth!,
  polar,
  polar!,
  qr,
  qr!,
  svd,
  svd!,
  svdvals,
  svdvals!,
  truncerr

using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit

for (f, f_full, f_compact) in (
  (:qr, :qr_full, :qr_compact),
  (:qr!, :qr_full!, :qr_compact!),
  (:lq, :lq_full, :lq_compact),
  (:lq!, :lq_full!, :lq_compact!),
)
  @eval begin
    function $f(A::AbstractMatrix; full::Bool=false, kwargs...)
      f = full ? $f_full : $f_compact
      return f(A; kwargs...)
    end
  end
end

for (eigen, eigh_full, eig_full, eigh_trunc, eig_trunc) in (
  (:eigen, :eigh_full, :eig_full, :eigh_trunc, :eig_trunc),
  (:eigen!, :eigh_full!, :eig_full!, :eigh_trunc!, :eig_trunc!),
)
  @eval begin
    function $eigen(A::AbstractMatrix; trunc=nothing, ishermitian=nothing, kwargs...)
      ishermitian = @something ishermitian LinearAlgebra.ishermitian(A)
      f = if !isnothing(trunc)
        ishermitian ? $eigh_trunc : $eig_trunc
      else
        ishermitian ? $eigh_full : $eig_full
      end
      return f(A; kwargs...)
    end
  end
end

for (eigvals, eigh_vals, eig_vals) in
    ((:eigvals, :eigh_vals, :eig_vals), (:eigvals!, :eigh_vals!, :eig_vals!))
  @eval begin
    function $eigvals(A::AbstractMatrix; ishermitian=nothing, kwargs...)
      ishermitian = @something ishermitian LinearAlgebra.ishermitian(A)
      f = (ishermitian ? $eigh_vals : $eig_vals)
      return f(A; kwargs...)
    end
  end
end

for (svd, svd_trunc, svd_full, svd_compact) in (
  (:svd, :svd_trunc, :svd_full, :svd_compact),
  (:svd!, :svd_trunc!, :svd_full!, :svd_compact!),
)
  @eval begin
    function $svd(A::AbstractMatrix; full::Bool=false, trunc=nothing, kwargs...)
      return if !isnothing(trunc)
        @assert !full "Specified both full and truncation, currently not supported"
        $svd_trunc(A; trunc, kwargs...)
      else
        (full ? $svd_full : $svd_compact)(A; kwargs...)
      end
    end
  end
end

for (svdvals, svd_vals) in ((:svdvals, :svd_vals), (:svdvals!, :svd_vals!))
  @eval begin
    function $svdvals(A::AbstractMatrix; ishermitian=nothing, kwargs...)
      return $svd_vals(A; kwargs...)
    end
  end
end

for (polar, left_polar, right_polar) in
    ((:polar, :left_polar, :right_polar), (:polar!, :left_polar!, :right_polar!))
  @eval begin
    function $polar(A::AbstractMatrix; side=:left, kwargs...)
      f = if side == :left
        $left_polar
      elseif side == :right
        $right_polar
      else
        throw(ArgumentError("`side=$side` not supported."))
      end
      return f(A; kwargs...)
    end
  end
end

for (orth, left_orth, right_orth) in
    ((:orth, :left_orth, :right_orth), (:orth!, :left_orth!, :right_orth!))
  @eval begin
    function $orth(A::AbstractMatrix; side=:left, kwargs...)
      f = if side == :left
        $left_orth
      elseif side == :right
        $right_orth
      else
        throw(ArgumentError("`side=$side` not supported."))
      end
      return f(A; kwargs...)
    end
  end
end

for (factorize, orth_f) in ((:factorize, :(MatrixAlgebra.orth)), (:factorize!, :orth!))
  @eval begin
    function $factorize(A::AbstractMatrix; orth=:left, kwargs...)
      f = if orth in (:left, :right)
        $orth_f
      else
        throw(ArgumentError("`orth=$orth` not supported."))
      end
      return f(A; side=orth, kwargs...)
    end
  end
end

using MatrixAlgebraKit: MatrixAlgebraKit, TruncationStrategy

struct TruncationError{T<:Real} <: TruncationStrategy
  atol::T
  rtol::T
  p::Int
end

"""
    truncerr(; atol::Real=0, rtol::Real=0, p::Int=2)

Create a truncation strategy for truncating such that the error in the factorization
is smaller than `max(atol, rtol * norm)`, where the error is determined using the `p`-norm.
"""
function truncerr(; atol::Real=0, rtol::Real=0, p::Int=2)
  return TruncationError(promote(atol, rtol)..., p)
end

function MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::TruncationError)
  Base.require_one_based_indexing(values)
  issorted(values; rev=true) || error("Not sorted.")
  # norm(values, p) ^ p
  normᵖ = sum(Base.Fix2(^, strategy.p) ∘ abs, values)
  ϵᵖ = max(strategy.atol ^ strategy.p, strategy.rtol ^ strategy.p * normᵖ)
  if ϵᵖ ≥ normᵖ
    return Base.OneTo(0)
  end
  truncerrᵖ = zero(real(eltype(values)))
  rank = length(values)
  for i in reverse(eachindex(values))
    truncerrᵖ += abs(values[i]) ^ strategy.p
    if truncerrᵖ ≥ ϵᵖ
      rank = i
      break
    end
  end
  return Base.OneTo(rank)
end

struct TruncationDegenerate{Strategy<:TruncationStrategy,T<:Real} <: TruncationStrategy
  strategy::Strategy
  atol::T
  rtol::T
end

"""
    truncdegen(trunc::TruncationStrategy; atol::Real=0, rtol::Real=0)

Modify a truncation strategy so that if the truncation falls within
a degenerate subspace, the entire subspace gets truncated as well.
`max(atol, rtol * σ₁)` where `σ₁` is the maximum value in the spectrum
being truncated.

For now, this truncation strategy assumes the spectrum being truncated
has already been reverse sorted and the strategy being wrapped
outputs a contiguous subset of values including the largest one. It
also only truncates for now, so may not respect if a minimum dimension
was requested in the strategy being wrapped. These restrictions may
be lifted in the future or provided through a different truncation strategy.
"""
function truncdegen(strategy::TruncationStrategy; atol::Real=0, rtol::Real=0)
  return TruncationDegenerate(strategy, promote(atol, rtol)...)
end

using MatrixAlgebraKit: findtruncated

function MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::TruncationDegenerate)
  Base.require_one_based_indexing(values)
  issorted(values; rev=true) || throw(ArgumentError("Values aren't reverse sorted."))
  indices_collection = findtruncated(values, strategy.strategy)
  indices = Base.OneTo(maximum(indices_collection))
  indices_collection == indices || throw(ArgumentError("Truncation must be a contiguous range."))
  if length(indices_collection) == length(values)
    # No truncation occured.
    return indices
  end
  # Value of the largest truncated value.
  val = values[last(indices) + 1]
  tol = max(strategy.atol, strategy.rtol * first(values))
  ind = last(indices)
  for i in reverse(Base.OneTo(last(indices)))
    if abs(values[i] - val) ≤ tol
      ind = i - 1
    else
      break
    end
  end
  return Base.OneTo(ind)
end

end
