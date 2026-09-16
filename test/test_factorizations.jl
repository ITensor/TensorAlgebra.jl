using LinearAlgebra: LinearAlgebra, Diagonal, I, diag, norm
using MatrixAlgebraKit: truncrank
using TensorAlgebra: TensorAlgebra, contract, contractalign, eig_full, eig_vals, eigh_full,
    eigh_vals, left_null, left_orth, left_polar, lq_compact, lq_full, qr_compact, qr_full,
    right_null, right_orth, right_polar, svd_compact, svd_full, svd_trunc, svd_vals
using Test: @test, @testset
using TestExtras: @constinferred

# Matricize without permuting: the identity bipermutation for a split after `k` dimensions.
splitperms(a, k) = (ntuple(identity, k), ntuple(i -> k + i, ndims(a) - k))

elts = (Float64, ComplexF64)

# QR Decomposition
# ----------------
@testset "Full QR ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:b, :a)
    labels_R = (:d, :c)

    Acopy = copy(A)
    Q, R = @constinferred qr_full(A, labels_A, labels_Q, labels_R)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
    @test A ≈ A′
    @test size(Q, 1) * size(Q, 2) == size(Q, 3) # Q is unitary

    Q, R = qr_full(A, (2, 1), (4, 3))
    @test A ≈ contractalign(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))

    Q, R = qr_full(A, Val(2))
    @test A ≈ contractalign((:a, :b, :c, :d), Q, (:a, :b, :q), R, (:q, :c, :d))
end

@testset "Compact QR ($T)" for T in elts
    A = randn(T, 2, 3, 4, 5) # compact only makes a difference for less columns
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:b, :a)
    labels_R = (:d, :c)

    Acopy = copy(A)
    Q, R = @constinferred qr_compact(A, labels_A, labels_Q, labels_R)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, Q, (labels_Q..., :q), R, (:q, labels_R...))
    @test A ≈ A′
    @test size(Q, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

# LQ Decomposition
# ----------------
@testset "Full LQ ($T)" for T in elts
    A = randn(T, 2, 3, 4, 5)
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:d, :c)
    labels_L = (:b, :a)

    Acopy = copy(A)
    L, Q = @constinferred lq_full(A, labels_A, labels_L, labels_Q)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
    @test A ≈ A′
    @test size(Q, 1) == size(Q, 2) * size(Q, 3) # Q is unitary

    L, Q = lq_full(A, (2, 1), (4, 3))
    @test A ≈ contractalign(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
end

@testset "Compact LQ ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2) # compact only makes a difference for less rows
    labels_A = (:a, :b, :c, :d)
    labels_Q = (:d, :c)
    labels_L = (:b, :a)

    Acopy = copy(A)
    L, Q = @constinferred lq_compact(A, labels_A, labels_L, labels_Q)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, L, (labels_L..., :q), Q, (:q, labels_Q...))
    @test A ≈ A′
    @test size(Q, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4)) # Q is unitary
end

# Eigenvalue Decomposition
# ------------------------
@testset "Eigenvalue decomposition ($T)" for T in elts
    A = randn(T, 4, 3, 4, 3) # needs to be square
    labels_A = (:a, :b, :c, :d)
    labels_V = (:b, :a)
    labels_V′ = (:d, :c)

    Acopy = copy(A)
    D, V = eig_full(A, labels_A, labels_V, labels_V′)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) == eltype(V) && eltype(D) <: Complex
    # `D` is returned bare (the spectrum over the internal bond), which is a `Diagonal`.
    @test D isa Diagonal

    AV = contractalign((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contractalign((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    Dvals = eig_vals(A, labels_A, labels_V, labels_V′)
    @test Dvals ≈ diag(D)
    @test eltype(Dvals) <: Complex
end

@testset "Hermitian eigenvalue decomposition ($T)" for T in elts
    A = randn(T, 12, 12)
    A = reshape(A + A', 4, 3, 4, 3)
    labels_A = (:a, :b, :c, :d)
    labels_V = (:b, :a)
    labels_V′ = (:d, :c)

    Acopy = copy(A)
    D, V = eigh_full(A, labels_A, labels_V, labels_V′)
    @test A == Acopy # should not have altered initial array
    @test eltype(D) <: Real
    @test eltype(V) == eltype(A)
    @test D isa Diagonal

    AV = contractalign((:a, :b, :D), A, labels_A, V, (labels_V′..., :D))
    VD = contractalign((:a, :b, :D), V, (labels_V..., :D′), D, (:D′, :D))
    @test AV ≈ VD

    Dvals = eigh_vals(A, labels_A, labels_V, labels_V′)
    @test Dvals ≈ diag(D)
    @test eltype(Dvals) <: Real
end

# Singular Value Decomposition
# ----------------------------
@testset "Full SVD ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_U = (:b, :a)
    labels_Vᴴ = (:d, :c)

    Acopy = copy(A)
    U, S, Vᴴ = @constinferred svd_full(A, labels_A, labels_U, labels_Vᴴ)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    @test size(U, 1) * size(U, 2) == size(U, 3) # U is unitary
    @test size(Vᴴ, 1) == size(Vᴴ, 2) * size(Vᴴ, 3) # V is unitary

    U, S, Vᴴ = svd_full(A, (2, 1), (4, 3))
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    @test A ≈ contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))

    U, S, Vᴴ = @constinferred svd_full(A, labels_A, labels_A, ())
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(Vᴴ, 1) == 1

    U, S, Vᴴ = @constinferred svd_full(A, labels_A, (), labels_A)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (:u,), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_A...))
    @test A ≈ A′
    @test size(U, 2) == 1
end

@testset "Compact SVD ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_U = (:b, :a)
    labels_Vᴴ = (:d, :c)

    Acopy = copy(A)
    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, labels_U, labels_Vᴴ)
    @test A == Acopy # should not have altered initial array
    @test S isa Diagonal
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test A ≈ A′
    k = min(size(S)...)
    @test size(U, 3) == k == size(Vᴴ, 1)

    Svals = @constinferred svd_vals(A, labels_A, labels_U, labels_Vᴴ)
    @test Svals ≈ diag(S)

    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, labels_A, ())
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_A..., :u), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v,))
    @test A ≈ A′
    @test size(U, ndims(U)) == 1 == size(Vᴴ, 1)

    U, S, Vᴴ = @constinferred svd_compact(A, labels_A, (), labels_A)
    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (:u,), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_A...))
    @test A ≈ A′
    @test size(U, 1) == 1 == size(Vᴴ, 1)
end

@testset "Truncated SVD ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_U = (:b, :a)
    labels_Vᴴ = (:d, :c)

    # test truncated SVD
    Acopy = copy(A)
    _, S_untrunc, _ = svd_compact(A, labels_A, labels_U, labels_Vᴴ)

    trunc = truncrank(size(S_untrunc, 1) - 1)
    U, S, Vᴴ, ϵ = @constinferred svd_trunc(A, labels_A, labels_U, labels_Vᴴ; trunc)

    @test A == Acopy # should not have altered initial array
    US, labels_US = contract(U, (labels_U..., :u), S, (:u, :v))
    A′ = contractalign(labels_A, US, labels_US, Vᴴ, (:v, labels_Vᴴ...))
    @test norm(A - A′) ≈ S_untrunc[end]
    @test size(S, 1) == size(S_untrunc, 1) - 1
    # `ϵ` is the 2-norm of the discarded singular values (here the single dropped value).
    @test ϵ ≈ S_untrunc[end]
end

@testset "Nullspace ($T)" for T in elts
    A = randn(T, 5, 4, 3, 2)
    labels_A = (:a, :b, :c, :d)
    labels_codomain = (:b, :a)
    labels_domain = (:d, :c)

    Acopy = copy(A)
    N = @constinferred left_null(A, labels_A, labels_codomain, labels_domain)
    @test A == Acopy # should not have altered initial array
    # N^ba_n' * A^ba_dc = 0
    NA = contractalign(
        (:n, labels_domain...),
        conj(N),
        (labels_codomain..., :n),
        A,
        labels_A
    )
    @test norm(NA) ≈ 0 atol = 1.0e-14
    NN =
        contractalign(
        (:n, :n′),
        conj(N),
        (labels_codomain..., :n),
        N,
        (labels_codomain..., :n′)
    )
    @test NN ≈ LinearAlgebra.I

    Nᴴ = @constinferred right_null(A, labels_A, labels_codomain, labels_domain)
    @test A == Acopy # should not have altered initial array
    # A^ba_dc * N^dc_n' = 0
    AN = contractalign(
        (labels_codomain..., :n),
        A,
        labels_A,
        conj(Nᴴ),
        (:n, labels_domain...)
    )
    @test norm(AN) ≈ 0 atol = 1.0e-14
    NN = contractalign((:n, :n′), Nᴴ, (:n, labels_domain...), Nᴴ, (:n′, labels_domain...))
end

@testset "Left polar ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_W = (:b, :a)
    labels_P = (:d, :c)

    Acopy = copy(A)
    W, P = left_polar(A, labels_A, labels_W, labels_P)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
    @test A ≈ A′
    @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

@testset "Right polar ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = copy(A)
    P, W = right_polar(A, labels_A, labels_P, labels_W)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
    @test A ≈ A′
    @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))
end

@testset "Left orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_W = (:b, :a)
    labels_P = (:d, :c)

    Acopy = copy(A)
    W, P = left_orth(A, labels_A, labels_W, labels_P)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
    @test A ≈ A′
    @test size(W, 3) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))

    W, P = left_orth(A, (2, 1), (4, 3))
    @test A ≈ contractalign(labels_A, W, (labels_W..., :w), P, (:w, labels_P...))
end

@testset "Right orth ($T)" for T in elts
    A = randn(T, 2, 2, 2, 2)
    labels_A = (:a, :b, :c, :d)
    labels_P = (:b, :a)
    labels_W = (:d, :c)

    Acopy = copy(A)
    P, W = right_orth(A, labels_A, labels_P, labels_W)
    @test A == Acopy # should not have altered initial array
    A′ = contractalign(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
    @test A ≈ A′
    @test size(W, 1) == min(size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))

    P, W = right_orth(A, (2, 1), (4, 3))
    @test A ≈ contractalign(labels_A, P, (labels_P..., :w), W, (:w, labels_W...))
end

# one (identity tensor)
# ---------------------
# An identity tensor matricized along its codomain/domain partition is the
# identity matrix.
@testset "one ($T)" for T in elts
    A = randn(T, 2, 3, 2, 3)
    labels_A = (:a, :b, :c, :d)
    labels_cod = (:a, :b)
    labels_dom = (:c, :d)

    Acopy = copy(A)
    Id = @constinferred TensorAlgebra.one(A, labels_A, labels_cod, labels_dom)
    @test A == Acopy # should not have altered initial array

    @test size(Id) == size(A)
    @test eltype(Id) === T

    @test TensorAlgebra.matricize(Id, splitperms(Id, 2)...) ≈ I

    # `Val`, perm, and label entries agree.
    @test TensorAlgebra.one(A, Val(2)) ≈ Id
    @test TensorAlgebra.one(A, (1, 2), (3, 4)) ≈ Id

    # Non-trivial codomain/domain partition: codomain (a, b) interleaved with
    # domain (c, d) in the input layout. The result is permuted into the
    # canonical (cod, dom) order before matricizing, and the matricized form
    # is again the identity matrix.
    B = randn(T, 2, 2, 2, 2)
    labels_B = (:a, :c, :b, :d)
    Id_perm = TensorAlgebra.one(B, labels_B, labels_cod, labels_dom)
    @test TensorAlgebra.matricize(Id_perm, splitperms(Id_perm, 2)...) ≈ I
    # Perm- and biperm-tuple forms agree with the label form.
    @test TensorAlgebra.one(B, (1, 3), (2, 4)) ≈ Id_perm

    # In-place `one!` fills the identity into its argument and returns it.
    C = randn(T, 2, 3, 2, 3)
    Cret = @constinferred TensorAlgebra.one!(C, Val(2))
    @test Cret === C
    @test TensorAlgebra.matricize(C, splitperms(C, 2)...) ≈ I
    @test C ≈ TensorAlgebra.one(A, Val(2))

    # `unmatricize!` scatters a fused matrix back into an existing array.
    D = randn(T, 2, 3, 2, 3)
    Dmat = TensorAlgebra.matricize(D, splitperms(D, 2)...)
    E = similar(D)
    Eret = TensorAlgebra.unmatricize!(E, Dmat, Val(2))
    @test Eret === E
    @test E ≈ D
end

# Trace
# -----
@testset "tr ($T)" for T in elts
    A = randn(T, 2, 3, 2, 3)
    m = reshape(A, 6, 6)
    # The labels, bi-permutation, and codomain-rank forms all agree with the matrix trace of
    # the matricized map.
    @test TensorAlgebra.tr(A, (:i, :j, :ip, :jp), (:i, :j), (:ip, :jp)) ≈
        LinearAlgebra.tr(m)
    @test TensorAlgebra.tr(A, (1, 2), (3, 4)) ≈ LinearAlgebra.tr(m)
    @test TensorAlgebra.tr(A, Val(2)) ≈ LinearAlgebra.tr(m)
end

# Permuted entry points
# ---------------------
# The permuted entry points must not mutate the caller's array, through the identity,
# permuted-copy, and codomain/domain-swap matricizations alike.
@testset "Permuted forms: input preserved, factors reconstruct ($T)" for T in elts
    A = randn(T, 2, 3, 4)
    Acopy = copy(A)
    for (perm_codomain, perm_domain) in
        (((1, 2), (3,)), ((3, 1), (2,)), ((3,), (1, 2)), ((2,), (3, 1)))
        k = length(perm_codomain)
        A_perm = TensorAlgebra.bipermutedims(A, perm_codomain, perm_domain)
        A_mat = TensorAlgebra.matricize(A_perm, splitperms(A_perm, k)...)
        for f in (qr_compact, lq_compact, left_orth, right_orth)
            X, Y = f(A, perm_codomain, perm_domain)
            @test TensorAlgebra.matricize(X, splitperms(X, k)...) *
                TensorAlgebra.matricize(Y, splitperms(Y, 1)...) ≈ A_mat
        end
        for f in (svd_compact, svd_trunc)
            U, S, Vᴴ = f(A, perm_codomain, perm_domain)
            U_mat = TensorAlgebra.matricize(U, splitperms(U, k)...)
            @test U_mat * S * TensorAlgebra.matricize(Vᴴ, splitperms(Vᴴ, 1)...) ≈ A_mat
            @test U_mat' * U_mat ≈ I
        end
        @test svd_vals(A, perm_codomain, perm_domain) ≈ LinearAlgebra.svdvals(A_mat)
        N_tensor = left_null(A, perm_codomain, perm_domain)
        N = TensorAlgebra.matricize(N_tensor, splitperms(N_tensor, k)...)
        @test norm(N' * A_mat) ≈ 0 atol = 1.0e-13
        @test N' * N ≈ I
        Nᴴ_tensor = right_null(A, perm_codomain, perm_domain)
        Nᴴ = TensorAlgebra.matricize(Nᴴ_tensor, splitperms(Nᴴ_tensor, 1)...)
        @test norm(A_mat * Nᴴ') ≈ 0 atol = 1.0e-13
        @test Nᴴ * Nᴴ' ≈ I
        @test A == Acopy
    end
    B = randn(T, 2, 3, 2, 3)
    Bcopy = copy(B)
    for (perm_codomain, perm_domain) in
        (((1, 2), (3, 4)), ((3, 4), (1, 2)), ((2, 3), (4, 1)))
        B_perm = TensorAlgebra.bipermutedims(B, perm_codomain, perm_domain)
        B_mat = Matrix(TensorAlgebra.matricize(B_perm, splitperms(B_perm, 2)...))
        D, V = eig_full(B, perm_codomain, perm_domain)
        V_mat = TensorAlgebra.matricize(V, splitperms(V, 2)...)
        @test B_mat * V_mat ≈ V_mat * D
        sortvals(v) = sort(v; by = x -> (real(x), imag(x)))
        @test sortvals(eig_vals(B, perm_codomain, perm_domain)) ≈
            sortvals(LinearAlgebra.eigvals(B_mat))
        @test TensorAlgebra.tr(B, perm_codomain, perm_domain) ≈ LinearAlgebra.tr(B_mat)
        @test B == Bcopy
    end
end

# A wrapper whose `matricize` reshapes the parent's buffer but declares nothing about it (no
# `Base.dataids` overload), so any ownership inference from aliasing checks misclassifies it.
module FactorizationMatricizeTestUtils
    using TensorAlgebra: TensorAlgebra as TA
    struct AliasingArray{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::P
    end
    Base.size(a::AliasingArray) = size(a.parent)
    function Base.getindex(a::AliasingArray{<:Any, N}, I::Vararg{Int, N}) where {N}
        return a.parent[I...]
    end
    struct AliasingMatricize <: TA.MatricizeStyle end
    TA.MatricizeStyle(::Type{<:AliasingArray}) = AliasingMatricize()
    # Delegate every hook to the dense style on the unwrapped parent, so the matricization
    # aliases exactly where a plain `Array`'s would.
    unwrap(a::AliasingArray) = a.parent
    unwrap(a::AbstractArray) = a
    function TA.is_output_view(
            ::typeof(TA.matricizeop), ::AliasingMatricize, op, a, perm_codomain, perm_domain
        )
        return TA.is_output_view(
            TA.matricizeop, TA.ReshapeMatricize(), op, unwrap(a), perm_codomain, perm_domain
        )
    end
    function TA.matricizeopview(
            ::AliasingMatricize, op, a, perm_codomain, perm_domain
        )
        return TA.matricizeopview(
            TA.ReshapeMatricize(), op, unwrap(a), perm_codomain, perm_domain
        )
    end
    function TA.allocate_output(
            ::typeof(TA.matricizeop), ::AliasingMatricize, op, a, perm_codomain, perm_domain
        )
        return TA.allocate_output(
            TA.matricizeop, TA.ReshapeMatricize(), op, unwrap(a), perm_codomain, perm_domain
        )
    end
    function TA.matricizeop!(
            dest, ::AliasingMatricize, op, a, perm_codomain, perm_domain
        )
        return TA.matricizeop!(
            dest, TA.ReshapeMatricize(), op, unwrap(a), perm_codomain, perm_domain
        )
    end
    function TA.unmatricize(::AliasingMatricize, m, axes_codomain, axes_domain)
        return AliasingArray(
            TA.unmatricize(TA.ReshapeMatricize(), m, axes_codomain, axes_domain)
        )
    end
end
using .FactorizationMatricizeTestUtils: AliasingArray

@testset "Aliasing matricize wrapper: input preserved ($f)" for f in
    (qr_compact, svd_compact)
    parent = randn(2, 3, 4)
    A = AliasingArray(parent)
    parent_copy = copy(parent)
    f(A, Val(1))
    @test parent == parent_copy
    f(A, (1,), (2, 3))
    @test parent == parent_copy
end

@testset "Integer eltype through the wrappers" begin
    parent = rand(-9:9, 2, 3, 4)
    A = AliasingArray(parent)
    parent_copy = copy(parent)
    Q, R = qr_compact(A, Val(1))
    @test parent == parent_copy
    Q_mat = TensorAlgebra.matricize(Q, splitperms(Q, 1)...)
    @test eltype(Q_mat) === Float64
    @test Q_mat * TensorAlgebra.matricize(R, splitperms(R, 1)...) ≈ reshape(parent, 2, 12)
    @test svd_vals(A, (1,), (2, 3)) ≈ LinearAlgebra.svdvals(reshape(float.(parent), 2, 12))
    @test parent == parent_copy
end
