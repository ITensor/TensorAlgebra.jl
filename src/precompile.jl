using LinearAlgebra: Adjoint, Transpose, mul!

for T in (Float64, ComplexF64)
    # The dense permute kernel.
    for N in 1:6
        SVT = SV.StridedView{T, N, Memory{T}, typeof(identity)}
        precompile(
            permutedimsconjadd_strided!,
            (SVT, Bool, SVT, NTuple{N, Int}, Bool, Bool)
        )
    end
    # The dense matrix products a contraction reaches.
    for A in (Matrix{T}, Adjoint{T, Matrix{T}}, Transpose{T, Matrix{T}}),
            B in (Matrix{T}, Adjoint{T, Matrix{T}}, Transpose{T, Matrix{T}})

        precompile(mul!, (Matrix{T}, A, B, Bool, Bool))
        precompile(*, (A, B))
    end
end
