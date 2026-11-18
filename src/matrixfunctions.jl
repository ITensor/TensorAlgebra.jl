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

# `matricize` may alias `a`, which is safe since the matrix functions do not mutate their input.
for f in MATRIX_FUNCTIONS
    @eval begin
        function $f(a, ndims_codomain::Val; kwargs...)
            return $f(a, identitybiperm(ndims_codomain, Val(ndims(a)))...; kwargs...)
        end
        function $f(a, perm_codomain, perm_domain; kwargs...)
            a_mat = matricize(a, perm_codomain, perm_domain)
            axes_codomain, axes_domain = bipartition_axes(
                map(i -> axes(a, i), (perm_codomain..., perm_domain...)),
                Val(length(perm_codomain))
            )
            fa_mat = Base.$f(a_mat; kwargs...)
            return unmatricize(fa_mat, axes_codomain, axes_domain)
        end
        function $f(a, labels_a, labels_codomain, labels_domain; kwargs...)
            perm_codomain, perm_domain =
                biperm(Tuple.((labels_a, labels_codomain, labels_domain))...)
            return $f(a, perm_codomain, perm_domain; kwargs...)
        end
    end
end
