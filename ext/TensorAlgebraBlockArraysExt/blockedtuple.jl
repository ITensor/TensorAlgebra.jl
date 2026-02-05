import BlockArrays as BA
import TensorAlgebra as TA

BA.blockfirsts(bt::TA.AbstractBlockTuple) = TA.blockfirsts(bt)
BA.blocklasts(bt::TA.AbstractBlockTuple) = TA.blocklasts(bt)
BA.blocklength(bt::TA.AbstractBlockTuple) = TA.blocklength(bt)
BA.blocklengths(bt::TA.AbstractBlockTuple) = TA.blocklengths(bt)
BA.blocklengths(type::Type{<:TA.AbstractBlockTuple}) = TA.blocklengths(type)
BA.blocks(bt::TA.AbstractBlockTuple) = TA.blocks(bt)

TA.Block(I::BA.Block) = TA.Block(I.n)
TA.BlockRange(I::BA.BlockRange) = TA.BlockRange(I.indices)
TA.BlockIndexRange(I::BA.BlockIndexRange) = TA.BlockIndexRange(TA.Block(I.block), I.indices)
Base.:(==)(I::BA.Block, J::TA.Block) = I.n == J.n
Base.:(==)(I::TA.Block, J::BA.Block) = I.n == J.n
Base.getindex(bt::TA.AbstractBlockTuple, I::BA.Block) = bt[TA.Block(I)]
Base.getindex(bt::TA.AbstractBlockTuple, I::BA.BlockIndexRange) = bt[TA.BlockIndexRange(I)]
Base.getindex(bt::TA.AbstractBlockTuple, I::BA.BlockRange{1}) = bt[TA.BlockRange(I)]

BA.blocklasts(r::TA.BlockedOneTo) = TA.blocklasts(r)
