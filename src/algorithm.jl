"""
    TensorAlgebra.select_algorithm(f, args...; alg = nothing, kwargs...)

Resolve the algorithm operation `f` should run with on `args`. An `alg` of `nothing` defers to
[`TensorAlgebra.default_algorithm`](@ref); anything else is validated and passed through.

This is the forward-facing layer the per-operation resolvers sit under, so a caller that does not
care which operation it is dispatching writes `select_algorithm(f, ...)` and a backend registers
its choice on `default_algorithm(f, ...)`.
"""
function select_algorithm(f, args...; alg = nothing, kwargs...)
    isnothing(alg) && return default_algorithm(f, args...; kwargs...)
    return select_algorithm_specified(f, alg, args...; kwargs...)
end

"""
    TensorAlgebra.default_algorithm(f, args...)
    TensorAlgebra.default_algorithm(f, argtypes::Type...)

The algorithm operation `f` runs with on `args` when the caller names none. The types form is the
registration point for a storage type; the values form defaults to it.

Each operation bridges to its own resolver, so `default_algorithm(contract, A1, A2)` is
[`TensorAlgebra.default_contract_algorithm`](@ref).
"""
function default_algorithm(f, args...; kwargs...)
    return default_algorithm(f, map(typeof, args)...; kwargs...)
end
function default_algorithm(f, argtypes::Type...; kwargs...)
    return throw(MethodError(default_algorithm, (f, argtypes...)))
end

# `alg` named something. A resolved algorithm object passes through; anything else is a caller
# error, reported against the operation rather than as a `MethodError` from inside the resolver.
function select_algorithm_specified(f, alg, args...; kwargs...)
    return throw(
        ArgumentError("`$alg` is not an algorithm for `$f`")
    )
end
