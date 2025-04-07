module MatrixAlgebra

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
      return f(A; kwargs...)
    end
  end
end

end
