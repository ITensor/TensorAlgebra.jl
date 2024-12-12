using TensorAlgebra: TensorAlgebra
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(TensorAlgebra, :DocTestSetup, :(using TensorAlgebra); recursive=true)

include("make_index.jl")

makedocs(;
  modules=[TensorAlgebra],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="TensorAlgebra.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/TensorAlgebra.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/TensorAlgebra.jl", devbranch="main", push_preview=true
)
