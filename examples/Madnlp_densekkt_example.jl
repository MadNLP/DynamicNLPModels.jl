using DynamicNLPModels, Random, LinearAlgebra, SparseArrays
using MadNLP, QuadraticModels, MadNLPGPU, CUDA, NLPModels

# Extend MadNLP functions
function MadNLP.jac_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, jac) where {T, V, M1<: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_jac)

    J = nlp.data.A
    copyto!(jac, J)
end

function MadNLP.hess_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, w1l, hess; obj_weight = 1.0) where {T, V, M1<: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_hess)
    H = nlp.data.H
    copyto!(hess, H)
end

# Time horizon
N  = 3

# generate random Q, R, A, and B matrices
Random.seed!(10)
Q_rand = Random.rand(2, 2)
Q = Q_rand * Q_rand' + I
R_rand   = Random.rand(1, 1)
R    = R_rand * R_rand' + I

A_rand = rand(2, 2)
A = A_rand * A_rand' + I
B = rand(2, 1)

# generate upper and lower bounds
sl = rand(2)
ul = fill(-15.0, 1)
su = sl .+ 4
uu = ul .+ 10
s0 = sl .+ 2

# Define K matrix for numerical stability of condensed problem
K  = - [1.41175 2.47819;] # found from MatrixEquations.jl; ared(A, B, 1, 1)


# Build model for 1 D heat transfer
lq_dense  = DenseLQDynamicModel(s0, A, B, Q, R, N; K = K, sl = sl, su = su, ul = ul, uu = uu)
lq_sparse = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, su = su, ul = ul, uu = uu)

# Solve the dense problem
dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=> LapackCPUSolver,
    :max_iter=> 50,
    :jacobian_constant=>true,
    :hessian_constant=>true,
    :lapack_algorithm=>MadNLP.CHOLESKY
)

d_ips = MadNLP.InteriorPointSolver(lq_dense, option_dict = dense_options)
sol_ref_dense = MadNLP.optimize!(d_ips)

# Solve the sparse problem
sparse_options = Dict{Symbol, Any}(
    :max_iter=>50,
    :jacobian_constant=>true,
    :hessian_constant=>true,
)

s_ips = MadNLP.InteriorPointSolver(lq_sparse, option_dict = sparse_options)
sol_ref_sparse = MadNLP.optimize!(s_ips)

# Solve the dense problem on the GPU
gpu_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
        :linear_solver=>LapackGPUSolver,
        :max_iter=>50,
        :jacobian_constant=>true,
        :hessian_constant=>true,
        :lapack_algorithm=>MadNLP.CHOLESKY
)

gpu_ips = MadNLPGPU.CuInteriorPointSolver(lq_dense, option_dict = gpu_options)
sol_ref_gpu = MadNLP.optimize!(gpu_ips)

println("States from dense problem on CPU are ", get_s(sol_ref_dense, lq_dense))
println("States from dense problem on GPU are ", get_s(sol_ref_gpu, lq_dense))
println("States from sparse problem on CPU are ", get_s(sol_ref_sparse, lq_sparse))
println()
println("Inputs from dense problem on CPU are ", get_u(sol_ref_dense, lq_dense))
println("Inputs from dense problem on GPU are ", get_u(sol_ref_gpu, lq_dense))
println("Inputs from sparse problem on CPU are ", get_u(sol_ref_sparse, lq_sparse))
