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

# The wrappers share the factorization machinery: `matricized` cores below consume the
# maybe-alias matricization read-only (the matrix functions allocate their own outputs), so
# the permuted forms skip the eager `bipermutedims` copy (see `factorizations.jl`).
for f in MATRIX_FUNCTIONS
    @eval begin
        function $f(style::MatricizeStyle, a, ndims_codomain::Val; kwargs...)
            a_mat = matricize(style, a, ndims_codomain)
            axes_codomain, axes_domain = bipartition_axes(axes(a), ndims_codomain)
            return matricized(
                $f, style, a_mat, isdetached(a_mat, a),
                axes_codomain, axes_domain; kwargs...
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
            a_mat = matricize_input(style, a, perm_codomain, perm_domain)
            axes_codomain, axes_domain = bipartition_axes(
                map(i -> axes(a, i), (perm_codomain..., perm_domain...)),
                Val(length(perm_codomain))
            )
            return matricized(
                $f, style, a_mat, isdetached(a_mat, a),
                axes_codomain, axes_domain; kwargs...
            )
        end
        function $f(
                a,
                perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
                kwargs...
            )
            return $f(MatricizeStyle(a), a, perm_codomain, perm_domain; kwargs...)
        end

        function matricized(
                ::typeof($f), style::MatricizeStyle, a_mat, owned::Bool,
                axes_codomain, axes_domain; kwargs...
            )
            fa_mat = Base.$f(a_mat; kwargs...)
            return unmatricize(style, fa_mat, axes_codomain, axes_domain)
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
