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
        function $f(a::AbstractArray, labels_a, labels_codomain, labels_domain; kwargs...)
            biperm = blockedperm_indexin(Tuple.((labels_a, labels_codomain, labels_domain))...)
            return $f(a, biperm; kwargs...)
        end
        function $f(a::AbstractArray, biperm::AbstractBlockPermutation{2}; kwargs...)
            a_mat = matricize(a, biperm)
            fa_mat = Base.$f(a_mat; kwargs...)
            return unmatricize(fa_mat, axes(a)[biperm])
        end
    end
end
