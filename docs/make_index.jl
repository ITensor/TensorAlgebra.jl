using Literate: Literate
using TensorAlgebra: TensorAlgebra

function ccq_logo(content)
  include_ccq_logo = """
    ```@raw html
    <img class="display-light-only" src="assets/CCQ.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo."/>
    <img class="display-dark-only" src="assets/CCQ-dark.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo."/>
    ```
    """
  content = replace(content, "{CCQ_LOGO}" => include_ccq_logo)
  return content
end

Literate.markdown(
  joinpath(pkgdir(TensorAlgebra), "examples", "README.jl"),
  joinpath(pkgdir(TensorAlgebra), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
  postprocess=ccq_logo,
)
