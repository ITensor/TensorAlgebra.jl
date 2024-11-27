using Literate: Literate
using TensorAlgebra: TensorAlgebra

Literate.markdown(
  joinpath(pkgdir(TensorAlgebra), "examples", "README.jl"),
  joinpath(pkgdir(TensorAlgebra), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
