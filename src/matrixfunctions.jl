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

# The matrix functions never mutate their input (they allocate their own outputs), so the
# permuted forms consume the maybe-alias `matricizeperm` matricization read-only, skipping
# the eager `bipermutedims` copy at the identity bipermutation.
for f in MATRIX_FUNCTIONS
    @eval begin
        function $f(style::MatricizeStyle, a, ndims_codomain::Val{K}; kwargs...) where {K}
            return $f(
                style, a,
                ntuple(identity, ndims_codomain),
                ntuple(i -> K + i, Val(ndims(a) - K));
                kwargs...
            )
        end
        function $f(a, ndims_codomain::Val; kwargs...)
            return $f(MatricizeStyle(a), a, ndims_codomain; kwargs...)
        end

        function $f(
                style::MatricizeStyle, a,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            a_mat = matricizeperm(style, a, perm_codomain, perm_domain)
            axes_codomain, axes_domain = bipartition_axes(
                map(i -> axes(a, i), (perm_codomain..., perm_domain...)),
                Val(length(perm_codomain))
            )
            fa_mat = Base.$f(a_mat; kwargs...)
            return unmatricize(style, fa_mat, axes_codomain, axes_domain)
        end
        function $f(
                a,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            return $f(MatricizeStyle(a), a, perm_codomain, perm_domain; kwargs...)
        end

        function $f(
                style::MatricizeStyle, a,
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
