using TensorAlgebra: TensorAlgebra, checked_project, checked_project_map,
    checked_projectto!, project, project_map, projectto!
using Test: @test, @test_throws, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)

# `projectto!` defaults to `copyto!` and so cannot itself produce a
# discrepancy that `checked_projectto!` would reject. To exercise the
# tolerance check we wrap the destination in a custom array type whose
# `projectto!` method rounds to one decimal place.
struct Rounded{T, A <: AbstractArray{T}} <: AbstractArray{T, 2}
    data::A
end
Base.size(r::Rounded) = size(r.data)
Base.getindex(r::Rounded, I...) = r.data[I...]
Base.setindex!(r::Rounded, v, I...) = (r.data[I...] = v; r)
function TensorAlgebra.projectto!(dest::Rounded, src::AbstractArray)
    dest.data .= round.(src; digits = 1)
    return dest
end

@testset "projectto!/project ($T)" for T in elts
    src = randn(T, 2, 3)

    # `projectto!` defaults to `copyto!`.
    dest = similar(src)
    @test projectto!(dest, src) === dest
    @test dest == src

    # `checked_projectto!` accepts when the projection is exact.
    dest2 = similar(src)
    @test checked_projectto!(dest2, src) === dest2
    @test dest2 == src

    # `checked_projectto!` rejects when the custom `projectto!` discards
    # information beyond `isapprox`'s tolerance.
    if T <: Real
        rough_src = randn(T, 2, 3)
        dest3 = Rounded(similar(rough_src))
        @test_throws InexactError checked_projectto!(
            dest3,
            rough_src;
            atol = 0.0,
            rtol = 0.0
        )
    end

    # `project_map` allocates an `(cod..., conj.(dom)...)`-shaped buffer and
    # projects `raw` into it.
    raw = randn(T, 2, 3, 2, 3)
    cod = (Base.OneTo(2), Base.OneTo(3))
    dom = (Base.OneTo(2), Base.OneTo(3))
    Mmap = project_map(raw, cod, dom)
    @test eltype(Mmap) === T
    @test size(Mmap) == (2, 3, 2, 3)
    @test Mmap == raw
    @test checked_project_map(raw, cod, dom) == raw

    # `project` forwards to `project_map`: the three-argument form takes the
    # explicit split, the two-argument form takes a flat list (empty domain).
    @test project(raw, cod, dom) == raw
    @test checked_project(raw, cod, dom) == raw

    flat = randn(T, 2, 3)
    M = project(flat, (Base.OneTo(2), Base.OneTo(3)))
    @test eltype(M) === T
    @test size(M) == (2, 3)
    @test M == flat
    @test checked_project(flat, (Base.OneTo(2), Base.OneTo(3))) == flat
end
