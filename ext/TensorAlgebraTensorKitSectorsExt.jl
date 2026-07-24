module TensorAlgebraTensorKitSectorsExt

using TensorAlgebra: TensorAlgebra
using TensorKitSectors: TensorKitSectors, Sector

# A sector's dual is its conjugate (charge conjugation), forwarded to TensorKitSectors' `dual`.
TensorAlgebra.dual(c::Sector) = TensorKitSectors.dual(c)

end
