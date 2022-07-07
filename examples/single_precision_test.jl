using Revise
using DynamicNLPModels
using MadNLP, LinearAlgebra, Random, SparseArrays, NLPModels


function MadNLP.jac_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, jac) where {T, V, M1<: AbstractMatrix{T}, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_jac)
    
    J = nlp.data.A
    copyto!(jac, J)
end

function MadNLP.hess_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, w1l, hess; obj_weight = 1.0) where {T, V, M1<: AbstractMatrix{T}, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_hess)
    H = nlp.data.H
    copyto!(hess, H)
end


ns = 5
nu = 5
N  = 5

Random.seed!(10)
Q_rand = Random.rand(ns, ns)
Q = Q_rand * Q_rand' + I
R_rand   = Random.rand(nu,nu)
R    = R_rand * R_rand' + I

A_rand = rand(ns, ns)
A = A_rand * A_rand' + I
B = rand(ns, nu)

# generate upper and lower bounds
sl = rand(ns)
ul = fill(-15.0, nu)
su = sl .+ 4
uu = ul .+ 10
s0 = sl .+ 2

lqdm64 = DenseLQDynamicModel(s0, A, B, Q, R, N)

function convert_precision(lqdm, T; dense=true)
    dnlp = lqdm.dynamic_data

    N  = dnlp.N
    A  = T.(dnlp.A)
    B  = T.(dnlp.B)
    Q  = T.(dnlp.Q)
    R  = T.(dnlp.R)
    s0 = T.(dnlp.s0)

    if dense
        return DenseLQDynamicModel(s0, A, B, Q, R, N)
    else
        return SparseLQDynamicModel(s0, A, B, Q, R, N)
    end
end

lqdm32 = convert_precision(lqdm64, Float32)
lqdm32s = convert_precision(lqdm64, Float32; dense=false)

# Solve the dense problem
dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=> LapackCPUSolver,
    :max_iter=> 200,
    :jacobian_constant=>true,
    :hessian_constant=>true,
)

d_ips = MadNLP.InteriorPointSolver(lqdm32, option_dict = dense_options)
sol_ref_dense = MadNLP.optimize!(d_ips)

# Solve the sparse problem
dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.SPARSE_KKT_SYSTEM,
    :linear_solver=> LapackCPUSolver,
    :max_iter=> 200,
    :jacobian_constant=>true,
    :hessian_constant=>true,
)

s_ips = MadNLP.InteriorPointSolver(lqdm32s, option_dict = dense_options)
sol_ref_sparse = MadNLP.optimize!(s_ips)

include("build_thinplate.jl")


ns = 5
nu = 5
N  = 10

function dfunc(i,j)
    return 100 * sin(2 * pi * (4 * i / N - 12 * j / ns)) + 400
end

d = [dfunc(i, j) for j in 1:ns, i in 1:(N + 1)]


dx = 0.1
dt = 0.1

# Build model for 1 D heat transfer
lq_dense  = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = true)
lq_sparse = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = false)

lqdm32 = convert_precision(lq_dense, Float32)

# Solve the dense problem
dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=> LapackCPUSolver,
    :max_iter=> 200,
    :jacobian_constant=>true,
    :hessian_constant=>true,
)

d_ips = MadNLP.InteriorPointSolver(lqdm32, option_dict = dense_options)
sol_ref_dense = MadNLP.optimize!(d_ips)
