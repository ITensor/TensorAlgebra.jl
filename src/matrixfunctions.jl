# TensorAlgebra version of matrix functions.
const MATRIX_FUNCTIONS = [
    :exp,
    :cis,
    :log,
    :sqrt,
    :cbrt,
    :cos,
    :sin,
    :tan,
    :csc,
    :sec,
    :cot,
    :cosh,
    :sinh,
    :tanh,
    :csch,
    :sech,
    :coth,
    :acos,
    :asin,
    :atan,
    :acsc,
    :asec,
    :acot,
    :acosh,
    :asinh,
    :atanh,
    :acsch,
    :asech,
    :acoth,
]

for f in MATRIX_FUNCTIONS
    @eval begin
        function $f(a::AbstractArray, ndims_codomain::Val; kwargs...)
            a_mat = matricize(a, ndims_codomain)
            fa_mat = Base.$f(a_mat; kwargs...)
            biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
            return unmatricize(fa_mat, axes(a)[biperm])
        end
        function $f(
                a::AbstractArray,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...,
            )
            a_perm = bipermutedims(a, perm_codomain, perm_domain)
            return $f(a_perm, Val(length(perm_codomain)); kwargs...)
        end
        function $f(a::AbstractArray, labels_a, labels_codomain, labels_domain; kwargs...)
            biperm = blockedperm_indexin(Tuple.((labels_a, labels_codomain, labels_domain))...)
            return $f(a, blocks(biperm)...; kwargs...)
        end
        function $f(a::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
            return $f(a, blocks(biperm)...; kwargs...)
        end
    end
end
