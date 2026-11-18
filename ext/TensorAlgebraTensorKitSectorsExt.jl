module TensorAlgebraTensorKitSectorsExt

using TensorAlgebra: TensorAlgebra
using TensorKitSectors: TensorKitSectors, Sector

TensorAlgebra.dual(c::Sector) = TensorKitSectors.dual(c)

end
