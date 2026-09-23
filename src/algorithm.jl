"""
    TensorAlgebra.AbstractAlgorithm

Supertype for the algorithm objects operations dispatch on. An operation's own supertype
subtypes this (for example `AbstractContractAlgorithm`), which is what makes an instance pass
through [`TensorAlgebra.select_algorithm`](@ref) unchanged.
"""
abstract type AbstractAlgorithm end

"""
    TensorAlgebra.select_algorithm(f, alg, args...)

Resolve the algorithm operation `f` should run with on `args`. An `alg` of `nothing` defers to
[`TensorAlgebra.default_algorithm`](@ref), an `AbstractAlgorithm` passes through unchanged, and
anything else is an error.

`alg` is positional so each operation can dispatch on the algorithm type. The user-facing entry
points take it as a keyword and hand it here.
"""
select_algorithm(f, ::Nothing, args...) = default_algorithm(f, args...)
select_algorithm(f, alg::AbstractAlgorithm, args...) = alg
# `alg` named something that is not an algorithm at all. Reported against the operation rather
# than as a `MethodError` from inside a resolver.
function select_algorithm(f, alg, args...)
    return throw(ArgumentError("`$alg` is not an algorithm for `$f`"))
end

"""
    TensorAlgebra.default_algorithm(f, args...)
    TensorAlgebra.default_algorithm(f, argtypes::Type...)

The algorithm operation `f` runs with on `args` when the caller names none. The types form is the
registration point for a storage type; the values form defaults to it.

A storage type registers its choice per operation, so a backend that contracts its own way
adds a method to `default_algorithm(contract!, A_dest, A1, A2)`.
"""
default_algorithm(f, args...) = default_algorithm(f, map(typeof, args)...)
function default_algorithm(f, argtypes::Type...)
    return throw(MethodError(default_algorithm, (f, argtypes...)))
end
