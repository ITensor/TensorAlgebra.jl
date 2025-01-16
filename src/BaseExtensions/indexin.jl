# `Base.indexin` doesn't handle tuples
indexin(x, y::AbstractArray) = Base.indexin(x, y)
indexin(x, y) = Base.indexin(x, collect(y))
indexin(x::Tuple, y::AbstractArray) = Tuple{Vararg{Any,length(x)}}(Base.indexin(x, y))
indexin(x::Tuple, y) = Tuple{Vararg{Any,length(x)}}(Base.indexin(x, collect(y)))
