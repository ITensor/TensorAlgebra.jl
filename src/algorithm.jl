"""
    TensorAlgebra.AbstractAlgorithm

Supertype for the algorithm objects operations dispatch on. An operation's own supertype
subtypes this (for example `AbstractContractAlgorithm`), which is what makes an instance pass
through [`TensorAlgebra.select_algorithm`](@ref) unchanged.
"""
abstract type AbstractAlgorithm end

"""
    TensorAlgebra.select_algorithm(f, alg, args::Tuple)

Resolve the algorithm operation `f` should run with on `args`. An `alg` of `nothing` defers to
[`TensorAlgebra.default_algorithm`](@ref), an `AbstractAlgorithm` passes through unchanged, and
anything else is an error.

`alg` is positional so each operation can dispatch on the algorithm type. The user-facing entry
points take it as a keyword and hand it here.

The selection-relevant arguments are packed into a tuple rather than spliced, so that the value
and type domains stay disjoint. `(Float64, Int)` is a pair of arguments that happen to be types,
while `Tuple{Float64, Int}` names the types of a pair of arguments.
"""
select_algorithm(f, ::Nothing, args::Tuple) = default_algorithm(f, args)
select_algorithm(f, alg::AbstractAlgorithm, args::Tuple) = alg
# `alg` named something that is not an algorithm at all. Reported against the operation rather
# than as a `MethodError` from inside a resolver.
function select_algorithm(f, alg, args::Tuple)
    return throw(ArgumentError("`$alg` is not an algorithm for `$f`"))
end

"""
    TensorAlgebra.default_algorithm(f, args::Tuple)
    TensorAlgebra.default_algorithm(f, Args::Type{<:Tuple})

The algorithm operation `f` runs with on `args` when the caller names none. The types form is the
registration point for a storage type, and the values form defaults to it.

A storage type registers its choice per operation, so a backend that contracts its own way adds a
method to `default_algorithm(contract!, ::Type{<:Tuple{A_dest, A1, A2}})`.
"""
default_algorithm(f, args::Tuple) = default_algorithm(f, typeof(args))
function default_algorithm(f, Args::Type{<:Tuple})
    return throw(MethodError(default_algorithm, (f, Args)))
end
