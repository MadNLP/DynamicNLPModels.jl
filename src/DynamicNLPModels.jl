module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays
import LinearOperators
import CUDA

import CUDA: CUBLAS
import SparseArrays: SparseMatrixCSC

export LQDynamicData, SparseLQDynamicModel, DenseLQDynamicModel, get_u, get_s, get_jacobian, add_jtsj!

include(joinpath("LinearQuadratic", "LinearQuadratic.jl"))
include(joinpath("LinearQuadratic", "sparse.jl"))
include(joinpath("LinearQuadratic", "dense.jl"))
include(joinpath("LinearQuadratic", "tools.jl"))

end # module
