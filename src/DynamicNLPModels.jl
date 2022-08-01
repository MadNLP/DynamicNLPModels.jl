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

include(joinpath("lq", "lq.jl"))
include(joinpath("lq", "sparse.jl"))
include(joinpath("lq", "dense.jl"))
include(joinpath("lq", "tools.jl"))
include(joinpath("nonlinear", "nonlinear.jl"))

end # module
