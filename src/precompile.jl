using LinearAlgebra: mul!
using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        for T in (Float64, ComplexF64)
            a = ones(T, (2, 2, 2))
            # Both operands are permuted before the matrix product.
            contract(a, (:a, :b, :c), a, (:d, :a, :e))
            # The dense permute kernel is specialized on the rank.
            for N in 1:6
                permutedims(ones(T, ntuple(Returns(2), N)), reverse(ntuple(identity, N)))
            end
            b = ones(T, (2, 2, 2, 2))
            # The matrix products a contraction reaches, depending on the operand layouts.
            m = ones(T, (2, 2))
            for m1 in (m, adjoint(m), transpose(m)), m2 in (m, adjoint(m), transpose(m))
                mul!(similar(m), m1, m2, true, false)
                mul!(allocate_output(*, m1, m2), m1, m2)
            end
            perm1, perm2 = (2, 1), (4, 3)
            matricize(b, perm1, perm2)
            svd_compact(b, perm1, perm2)
            svd_trunc(b, perm1, perm2; trunc = MatrixAlgebraKit.truncrank(2))
            qr_compact(b, perm1, perm2)
            lq_compact(b, perm1, perm2)
            eigh_full(b, perm1, perm2)
        end
    end
end
