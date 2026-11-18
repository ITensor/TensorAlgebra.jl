using LinearAlgebra: mul!
using TupleTools: TupleTools

# Matricize both operands, multiply, and write the product into the destination.
function contractpermconjadd!(
        ::MatricizeContract,
        a_dest::AbstractArray, biperm_dest_codomain, biperm_dest_domain,
        isconj1::Bool, a1::AbstractArray, biperm1_codomain, biperm1_domain,
        isconj2::Bool, a2::AbstractArray, biperm2_codomain, biperm2_domain,
        α::Number, β::Number
    )
    biperm_dest = (biperm_dest_codomain..., biperm_dest_domain...)
    TupleTools.isperm(biperm_dest) ||
        throw(ArgumentError("destination permutation is not a permutation"))
    invperm_codomain, invperm_domain =
        bipartition(TupleTools.invperm(biperm_dest), Val(length(biperm1_codomain)))
    check_input(
        contractpermconjadd!,
        a_dest, invperm_codomain, invperm_domain,
        isconj1, a1, biperm1_codomain, biperm1_domain,
        isconj2, a2, biperm2_codomain, biperm2_domain
    )
    (; swap, contracted) = matricized_layout(
        a_dest, invperm_codomain, invperm_domain,
        isconj1, a1, biperm1_codomain, biperm1_domain,
        isconj2, a2, biperm2_codomain, biperm2_domain
    )
    contracted1, contracted2 = contracted
    if swap
        _contractpermconjadd_matricized!(
            a_dest, invperm_domain, invperm_codomain,
            isconj2, a2, biperm2_domain, contracted2,
            isconj1, a1, contracted1, biperm1_codomain,
            α, β
        )
    else
        _contractpermconjadd_matricized!(
            a_dest, invperm_codomain, invperm_domain,
            isconj1, a1, biperm1_codomain, contracted1,
            isconj2, a2, contracted2, biperm2_domain,
            α, β
        )
    end
    return a_dest
end

# The layout of a matricized contraction that copies the fewest elements, to matricize the
# operands and to write the destination. The product is `a1 * a2`, or with `swap` the transposed
# `a2 * a1`, and the contracted legs are in `a1`'s or in `a2`'s storage order. Inlined, since the
# call alone measurably slows down small contractions.
@inline function matricized_layout(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1, a1, open1, contracted1,
        isconj2, a2, contracted2, open2
    )
    p = TupleTools.sortperm(contracted2)
    contracted1_alt = map(i -> contracted1[i], p)
    contracted2_alt = map(i -> contracted2[i], p)
    # `a1 * a2` with the contracted legs in `a1`'s and then in `a2`'s order, followed by `a2 * a1`
    # likewise.
    splits1 = (
        (open1, contracted1), (open1, contracted1_alt),
        (contracted1, open1), (contracted1_alt, open1),
    )
    splits2 = (
        (contracted2, open2), (contracted2_alt, open2),
        (open2, contracted2), (open2, contracted2_alt),
    )
    split_dest = (perm_dest_codomain, perm_dest_domain)
    split_dest_swapped = (perm_dest_domain, perm_dest_codomain)
    splits_dest = (split_dest, split_dest, split_dest_swapped, split_dest_swapped)
    costs = map(splits1, splits2, splits_dest) do split1, split2, split_dest
        return cost(matricizeconj, isconj1, a1, split1...) +
            cost(matricizeconj, isconj2, a2, split2...) +
            cost(unmatricizeadd!, a_dest, split_dest...)
    end
    i = argmin(costs)
    contracted = isodd(i) ? (contracted1, contracted2) : (contracted1_alt, contracted2_alt)
    return (; swap = i > 2, contracted)
end

# The number of elements copied to matricize `a`, or to write the destination, through one split.
# Each sees one array and is kept out of line, so it compiles per array type rather than per pair
# of operands.
@noinline cost(::typeof(matricizeconj), isconj, a, perm_codomain, perm_domain) =
    isoutputview(matricizeconj, isconj, a, perm_codomain, perm_domain) ? 0 : length(a)
# Only a plain (not transposed) view is written through, since `mul!` into a lazy `transpose` does
# not use BLAS.
@noinline function cost(::typeof(unmatricizeadd!), a_dest, perm_codomain, perm_domain)
    isidentitybiperm(perm_codomain, perm_domain) &&
        isoutputview(matricizeconj, false, a_dest, perm_codomain, perm_domain) &&
        return 0
    return length(a_dest)
end

# `a_dest = β * a_dest + α * m1 * m2` for the operands matricized as `m1` and `m2`, with the
# product's rows and columns placed in `a_dest` by `(perm_dest_codomain, perm_dest_domain)`.
function _contractpermconjadd_matricized!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α::Number, β::Number
    )
    a1_mat = matricizeconj(isconj1, a1, perm1_codomain, perm1_domain)
    a2_mat = matricizeconj(isconj2, a2, perm2_codomain, perm2_domain)
    if isidentitybiperm(perm_dest_codomain, perm_dest_domain) &&
            isoutputview(matricizeconj, false, a_dest, perm_dest_codomain, perm_dest_domain)
        a_dest_mat = matricizeconjview(false, a_dest, perm_dest_codomain, perm_dest_domain)
        # The matricizations are inferred as a `Union` of `Matrix`, `Transpose` and `Adjoint`, which
        # would make the compiler analyze the matrix product on that `Union`. Dispatching at run
        # time reaches the method for the concrete types, which is usually already compiled.
        Base.inferencebarrier(mul!)(a_dest_mat, a1_mat, a2_mat, α, β)
    else
        Base.inferencebarrier(unmatricizemuladd!)(
            a_dest, a1_mat, a2_mat, perm_dest_codomain, perm_dest_domain, α, β
        )
    end
    return a_dest
end

# `a_dest = β * a_dest + α * unmatricize(m1 * m2)`.
function unmatricizemuladd!(a_dest, m1, m2, perm_codomain, perm_domain, α, β)
    unmatricizeadd!(a_dest, m1 * m2, perm_codomain, perm_domain, α, β)
    return nothing
end
