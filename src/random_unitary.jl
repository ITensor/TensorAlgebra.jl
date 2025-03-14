using Random: Random, AbstractRNG

function random_unitary(
  rng::AbstractRNG, elt::Type, ax::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}
)
  ax_fused = ⊗(ax...)
  a_fused = random_unitary(rng, elt, ax_fused)
  return splitdims(a_fused, dual.(ax), ax)
end

# Copy of `Base.to_dim`:
# https://github.com/JuliaLang/julia/blob/1431bec1bcd205f181ca2a3f1c314247b64076df/base/array.jl#L439-L440
to_dim(d::Integer) = d
to_dim(d::Base.OneTo) = last(d)

# Matrix version.
function random_unitary(rng::AbstractRNG, elt::Type, ax::Tuple{AbstractUnitRange})
  return random_unitary(rng, elt, map(to_dim, ax))
end

using MatrixAlgebraKit: qr_full!
function random_unitary(rng::AbstractRNG, elt::Type, dims::Tuple{Integer})
  Q, _ = qr_full!(randn(rng, elt, (dims..., dims...)); positive=true)
  return Q
end

# Canonicalizing other kinds of inputs.
function random_unitary(
  rng::AbstractRNG, elt::Type, dims::Tuple{Vararg{Union{AbstractUnitRange,Integer}}}
)
  return random_unitary(Random.default_rng(), elt, map(to_axis, dims))
end
function random_unitary(elt::Type, dims::Tuple{Vararg{Union{AbstractUnitRange,Integer}}})
  return random_unitary(Random.default_rng(), elt, dims)
end
function random_unitary(
  rng::AbstractRNG, elt::Type, dims::Union{AbstractUnitRange,Integer}...
)
  return random_unitary(rng, elt, dims)
end
function random_unitary(elt::Type, dims::Union{AbstractUnitRange,Integer}...)
  return random_unitary(Random.default_rng(), elt, dims)
end
function random_unitary(rng::AbstractRNG, dims::Union{AbstractUnitRange,Integer}...)
  return random_unitary(rng, Float64, dims)
end
function random_unitary(dims::Union{AbstractUnitRange,Integer}...)
  return random_unitary(Random.default_rng(), Float64, dims)
end
