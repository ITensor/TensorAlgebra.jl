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
        function $f(
                a::AbstractArray,
                codomain_length::Val, domain_length::Val;
                kwargs...,
            )
            a_mat = matricize(a, codomain_length, domain_length)
            fa_mat = Base.$f(a_mat; kwargs...)
            biperm = blockedtrivialperm((codomain_length, domain_length))
            return unmatricize(fa_mat, axes(a)[biperm])
        end
        function $f(
                a::AbstractArray,
                codomain_perm::Tuple{Vararg{Int}}, domain_perm::Tuple{Vararg{Int}};
                kwargs...,
            )
            a_perm = bipermutedims(a, codomain_perm, domain_perm)
            return $f(a_perm, Val(length(codomain_perm)), Val(length(domain_perm)); kwargs...)
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
