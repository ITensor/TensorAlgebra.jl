"""
    TensorAlgebra.AbstractAlgorithm

Supertype for algorithm objects that select how an operation is performed, for example
subtypes of `ContractAlgorithm`. An instance passes through
[`TensorAlgebra.select_algorithm`](@ref) unchanged.
"""
abstract type AbstractAlgorithm end

"""
    TensorAlgebra.select_algorithm(f, alg, args::Tuple)

Resolve the algorithm operation `f` should run with on `args`. An `alg` of `nothing` defers to
[`TensorAlgebra.default_algorithm`](@ref), an `AbstractAlgorithm` passes through unchanged, and
anything else is an error.
"""
select_algorithm(f, ::Nothing, args::Tuple) = default_algorithm(f, args)
select_algorithm(f, alg::AbstractAlgorithm, args::Tuple) = alg
function select_algorithm(f, alg, args::Tuple)
    return throw(ArgumentError("`$alg` is not an algorithm for `$f`"))
end

"""
    TensorAlgebra.default_algorithm(f, args::Tuple)
    TensorAlgebra.default_algorithm(f, Args::Type{<:Tuple})

The algorithm operation `f` runs with on `args` when the caller names none. An array type sets
its default by overloading the type form, for example
`default_algorithm(contract!, ::Type{<:Tuple{A_dest, A1, A2}})`.
"""
default_algorithm(f, args::Tuple) = default_algorithm(f, typeof(args))
function default_algorithm(f, Args::Type{<:Tuple})
    return throw(MethodError(default_algorithm, (f, Args)))
end
