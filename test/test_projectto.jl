using TensorAlgebra: TensorAlgebra, is_projected, project, project!, projectto!, tryproject,
    unchecked_project
using Test: @test, @test_throws, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)

# A dense projection is exact, so it cannot itself trip the `is_projected` check. To
# exercise the checked verbs generically we wrap the source in a custom array type whose
# `projectto!` method rounds to one decimal place, discarding information.
struct Rounded{T, A <: AbstractArray{T}} <: AbstractArray{T, 2}
    data::A
end
Base.size(r::Rounded) = size(r.data)
Base.getindex(r::Rounded, I...) = r.data[I...]
function TensorAlgebra.projectto!(dest::AbstractArray, src::Rounded)
    dest .= round.(src.data; digits = 1)
    return dest
end

@testset "projectto!/project ($T)" for T in elts
    src = randn(T, 2, 3)

    # `projectto!` is the in-place fill primitive and defaults to `copyto!`.
    dest = similar(src)
    @test projectto!(dest, src) === dest
    @test dest == src

    # `project!` is its checked sibling (as `copy!` is to `copyto!`).
    dest1 = similar(src)
    @test project!(dest1, src) === dest1
    @test dest1 == src

    # `project` projects into a `(cod..., conj.(dom)...)`-shaped destination (allocated by
    # `allocate_project`, filled by `projectto!`) and verifies via `is_projected` that
    # nothing was discarded. A dense projection is exact, so it always accepts here; the
    # symmetry-rejection cases are exercised on the GradedArrays and TensorKit backends.
    raw = randn(T, 2, 3, 2, 3)
    cod = (Base.OneTo(2), Base.OneTo(3))
    dom = (Base.OneTo(2), Base.OneTo(3))
    M = project(raw, cod, dom)
    @test eltype(M) === T
    @test size(M) == (2, 3, 2, 3)
    @test M == raw

    # `unchecked_project` is the same projection without the check.
    @test unchecked_project(raw, cod, dom) == raw

    # the two-argument forms take a flat list of axes (empty domain)
    flat = randn(T, 2, 3)
    Mflat = project(flat, (Base.OneTo(2), Base.OneTo(3)))
    @test eltype(Mflat) === T
    @test size(Mflat) == (2, 3)
    @test Mflat == flat
    @test unchecked_project(flat, (Base.OneTo(2), Base.OneTo(3))) == flat

    # A lossy `projectto!` (the `Rounded` fixture) flips the outcome of each verb:
    # `unchecked_project` silently returns the truncated result, `is_projected` reports
    # the discard, `project` throws, and `tryproject` gives `nothing`.
    if T <: Real
        rough = Rounded(randn(T, 2, 3))
        lossy = unchecked_project(rough, (Base.OneTo(2), Base.OneTo(3)))
        @test lossy == round.(rough.data; digits = 1)
        @test !is_projected(lossy, rough; atol = 0, rtol = 0)
        @test_throws InexactError project(
            rough, (Base.OneTo(2), Base.OneTo(3)); atol = 0, rtol = 0
        )
        @test isnothing(
            tryproject(rough, (Base.OneTo(2), Base.OneTo(3)); atol = 0, rtol = 0)
        )
        @test_throws InexactError project!(similar(rough.data), rough; atol = 0, rtol = 0)
    end
end

@testset "tryproject ($T)" for T in elts
    # `tryproject` is the nullable sibling of `project`: same projection and check, but
    # `nothing` instead of an `InexactError` when weight is discarded. A dense projection
    # is exact, so it always succeeds here; the failure cases are exercised on symmetric
    # backends (see the GradedArrays and TensorKit tests) and via the `Rounded` fixture.
    raw = randn(T, 2, 3)
    t = tryproject(raw, (Base.OneTo(2), Base.OneTo(3)))
    @test t == raw
    @test tryproject(raw, (Base.OneTo(2),), (Base.OneTo(3),)) == raw
end

@testset "project pads trailing length-1 axes ($T)" for T in elts
    # A caller may pass the dense data over the non-trivial axes and let `project`
    # supply the length-1 axes a split introduces (e.g. an auxiliary flux-canceling
    # leg on a symmetric state); `size(raw, d)` is 1 past `ndims(raw)`, matching
    # `raw[.., 1:1]` slicing.
    flat = randn(T, 2, 3)

    # a trailing length-1 axis absent from `raw`'s rank
    M = project(flat, (Base.OneTo(2), Base.OneTo(3), Base.OneTo(1)))
    @test size(M) == (2, 3, 1)
    @test vec(M) == vec(flat)

    # the length-1 axis may sit in the domain half of an explicit split
    Msplit = project(flat, (Base.OneTo(2), Base.OneTo(3)), (Base.OneTo(1),))
    @test size(Msplit) == (2, 3, 1)
    @test vec(Msplit) == vec(flat)
end
