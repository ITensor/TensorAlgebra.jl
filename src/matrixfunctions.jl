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
        function $f(style::FusionStyle, a, ndims_codomain::Val; kwargs...)
            a_mat = matricize(style, a, ndims_codomain)
            fa_mat = Base.$f(a_mat; kwargs...)
            codomain_axes, domain_axes = bipartition(axes(a), ndims_codomain)
            return unmatricize(style, fa_mat, codomain_axes, domain_axes)
        end
        function $f(a, ndims_codomain::Val; kwargs...)
            return $f(FusionStyle(a), a, ndims_codomain; kwargs...)
        end

        function $f(
                style::FusionStyle, a,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            a_perm = bipermutedims(a, perm_codomain, perm_domain)
            return $f(style, a_perm, Val(length(perm_codomain)); kwargs...)
        end
        function $f(
                a,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            a_perm = bipermutedims(a, perm_codomain, perm_domain)
            return $f(a_perm, Val(length(perm_codomain)); kwargs...)
        end

        function $f(
                style::FusionStyle, a,
                labels_a, labels_codomain, labels_domain; kwargs...
            )
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_a, labels_codomain, labels_domain))...)
            return $f(style, a, perm_codomain, perm_domain; kwargs...)
        end
        function $f(
                a,
                labels_a, labels_codomain, labels_domain; kwargs...
            )
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_a, labels_codomain, labels_domain))...)
            return $f(a, perm_codomain, perm_domain; kwargs...)
        end
    end
end
