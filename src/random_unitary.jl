using MatrixAlgebraKit: qr_full!
using Random: Random, AbstractRNG, randn!

function square_zero_map(elt::Type, ax::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}})
  return zeros(elt, (ax..., ax...))
end

using EllipsisNotation: : .. function random_unitary!(rng::AbstractRNG, a::AbstractArray)
  @assert iseven(ndims(a))
  ndims_codomain = ndims(a) ÷ 2
  biperm = blockedperm(ntuple(identity, ndims(a)), (ndims_codomain, ndims_codomain))
  a_mat = fusedims(a, biperm)
  r_mat = random_unitary!(rng, a_mat)
  splitdims!(a, r_mat, biperm)
  return a
end

function random_unitary!(rng::AbstractRNG, a::AbstractMatrix)
  a_r = randn!(rng, a)
  Q, _ = qr_full!(randn!(rng, a); positive=true)
  return Q
end

function random_unitary(
  rng::AbstractRNG, elt::Type, ax::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}
)
  return random_unitary!(rng, square_zero_map(elt, ax))
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
