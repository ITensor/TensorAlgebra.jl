using Test: @test, @testset, @inferred
using TensorOperations: @tensor, ncon, tensorcontract
using TensorAlgebra: Matricize

@testset "tensorcontract" begin
  A = randn(Float64, (3, 20, 5, 3, 4))
  B = randn(Float64, (5, 6, 20, 3))
  C1 = @inferred tensorcontract(
    A, ((1, 4, 5), (2, 3)), false, B, ((3, 1), (2, 4)), false, ((1, 5, 3, 2, 4), ()), 1.0
  )
  C2 = @inferred tensorcontract(
    A,
    ((1, 4, 5), (2, 3)),
    false,
    B,
    ((3, 1), (2, 4)),
    false,
    ((1, 5, 3, 2, 4), ()),
    1.0,
    Matricize(),
  )
  @test C1 ≈ C2
end

elts = (Float32, Float64, ComplexF32, ComplexF64)

@testset "tensor network examples ($T)" for T in elts
  D1, D2, D3 = 30, 40, 20
  d1, d2 = 2, 3
  A1 = rand(T, D1, d1, D2) .- 1//2
  A2 = rand(T, D2, d2, D3) .- 1//2
  rhoL = rand(T, D1, D1) .- 1//2
  rhoR = rand(T, D3, D3) .- 1//2
  H = rand(T, d1, d2, d1, d2) .- 1//2

  @tensor HrA12[a, s1, s2, c] :=
    rhoL[a, a'] * A1[a', t1, b] * A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]
  @tensor backend = Matricize() HrA12′[a, s1, s2, c] :=
    rhoL[a, a'] * A1[a', t1, b] * A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]

  @test HrA12 ≈ HrA12′
  @test HrA12 ≈ ncon(
    [rhoL, H, A2, rhoR, A1],
    [[-1, 1], [-2, -3, 4, 5], [2, 5, 3], [3, -4], [1, 4, 2]];
    backend=Matricize(),
  )
  E = @tensor rhoL[a', a] *
    A1[a, s, b] *
    A2[b, s', c] *
    rhoR[c, c'] *
    H[t, t', s, s'] *
    conj(A1[a', t, b']) *
    conj(A2[b', t', c'])
  @test E ≈ @tensor backend = Matricize() rhoL[a', a] *
    A1[a, s, b] *
    A2[b, s', c] *
    rhoR[c, c'] *
    H[t, t', s, s'] *
    conj(A1[a', t, b']) *
    conj(A2[b', t', c'])
end

function generate_random_network(
  num_contracted_inds, num_open_inds, max_dim, max_ind_per_tensor
)
  contracted_indices = repeat(collect(1:num_contracted_inds), 2)
  open_indices = collect(1:num_open_inds)
  dimensions = [
    repeat(rand(1:max_dim, num_contracted_inds), 2)
    rand(1:max_dim, num_open_inds)
  ]

  sizes = Vector{Int64}[]
  indices = Vector{Int64}[]

  while !isempty(contracted_indices) || !isempty(open_indices)
    num_inds = rand(
      1:min(max_ind_per_tensor, length(contracted_indices) + length(open_indices))
    )

    cur_inds = Int64[]
    cur_dims = Int64[]

    for _ in 1:num_inds
      curind_index = rand(1:(length(contracted_indices) + length(open_indices)))

      if curind_index <= length(contracted_indices)
        push!(cur_inds, contracted_indices[curind_index])
        push!(cur_dims, dimensions[curind_index])
        deleteat!(contracted_indices, curind_index)
        deleteat!(dimensions, curind_index)
      else
        tind = curind_index - length(contracted_indices)
        push!(cur_inds, -open_indices[tind])
        push!(cur_dims, dimensions[curind_index])
        deleteat!(open_indices, tind)
        deleteat!(dimensions, curind_index)
      end
    end

    push!(sizes, cur_dims)
    push!(indices, cur_inds)
  end
  return sizes, indices
end

@testset "random contractions" begin
  MAX_CONTRACTED_INDICES = 10
  MAX_OPEN_INDICES = 5
  MAX_DIM = 5
  MAX_IND_PER_TENS = 3
  NUM_TESTS = 10

  for _ in 1:NUM_TESTS
    sizes, indices = generate_random_network(rand(1:MAX_CONTRACTED_INDICES), rand(1:MAX_OPEN_INDICES), MAX_DIM, MAX_IND_PER_TENS)
    tensors = map(splat(randn), sizes)
    result1 = ncon(tensors, indices)
    result2 = ncon(tensors, indices; backend=Matricize())
    @test result1 ≈ result2
  end
end
