module TensorAlgebraTensorKitSectorsExt

using TensorAlgebra: TensorAlgebra
using TensorKitSectors: TensorKitSectors, Sector

# The dual of a sector is its conjugate (charge conjugation), which is TensorKitSectors' own.
TensorAlgebra.dual(c::Sector) = TensorKitSectors.dual(c)

end
