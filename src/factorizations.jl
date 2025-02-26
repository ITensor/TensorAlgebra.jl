using ArrayLayouts: LayoutMatrix
using LinearAlgebra: LinearAlgebra, Diagonal

function qr(a::AbstractArray, biperm::AbstractBlockPermutation{2})
  a_matricized = fusedims(a, biperm)
  # TODO: Make this more generic, allow choosing thin or full,
  # make sure this works on GPU.
  q_fact, r_matricized = LinearAlgebra.qr(a_matricized)
  q_matricized = typeof(a_matricized)(q_fact)
  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_q = tuplemortar((axes_codomain, (axes(q_matricized, 2),)))
  axes_r = tuplemortar(((axes(r_matricized, 1),), axes_domain))
  q = splitdims(q_matricized, axes_q)
  r = splitdims(r_matricized, axes_r)
  return q, r
end

function qr(a::AbstractArray, labels_a, labels_codomain, labels_domain)
  # TODO: Generalize to conversion to `Tuple` isn't needed.
  return qr(
    a, blockedperm_indexin(Tuple(labels_a), Tuple(labels_codomain), Tuple(labels_domain))
  )
end

function svd(a::AbstractArray, biperm::AbstractBlockPermutation{2})
  a_matricized = fusedims(a, biperm)
  usv_matricized = LinearAlgebra.svd(a_matricized)
  u_matricized = usv_matricized.U
  s_diag = usv_matricized.S
  v_matricized = usv_matricized.Vt
  axes_codomain, axes_domain = blockpermute(axes(a), biperm)
  axes_u = tuplemortar((axes_codomain, (axes(u_matricized, 2),)))
  axes_v = tuplemortar(((axes(v_matricized, 1),), axes_domain))
  u = splitdims(u_matricized, axes_u)
  # TODO: Use `DiagonalArrays.diagonal` to make it more general.
  s = Diagonal(s_diag)
  v = splitdims(v_matricized, axes_v)
  return u, s, v
end

function svd(a::AbstractArray, labels_a, labels_codomain, labels_domain)
  return svd(
    a, blockedperm_indexin(Tuple(labels_a), Tuple(labels_codomain), Tuple(labels_domain))
  )
end
