using Revise
using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU, CUDA, DelimitedFiles, SparseArrays, MKLSparse
include("build_thinplate.jl")


# Define attributes of 1 D heat transfer model
N  = 50 # time steps
ns = 50 # states
nu = 10 # inputs
nc = 10 # constraints

function dfunc(i,j)
    return 100 * sin(2 * pi * (4 * i / N - 12 * j / ns)) + 400
end

d = [dfunc(i, j) for i in 1:(N + 1), j in 1:ns]


dx = 0.1
dt = 0.1

K = 1.0 * Matrix(I, nu, ns)

S = -.001 * Matrix(I, ns, nu)

E = rand(nc, ns)
F = rand(nc, nu)

gl_val = -140 * 20.
gu_val = 140 * 20 + 450 * 20.

gl = fill(gl_val, nc)
gu = fill(gu_val, nc)

gl .*= rand(.8:.00001:1, nc)
gu .*= rand(.8:.00001:1, nc)

lq_dense_KSEF  = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu)
lq_sparse_KSEF = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = false, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu)

lq_dense_KS    = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S)
lq_sparse_KS   = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = false, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S)


madnlp_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=>MadNLPLapackGPU,
    :jacobian_constant=>true,
    :hessian_constant=>true,
)

linear_solver_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=>MadNLPLapackGPU,
    :jacobian_constant=>true,
    :hessian_constant=>true,
    :lapackgpu_algorithm => MadNLPLapackGPU.BUNCHKAUFMAN,
)

TKKTGPU = MadNLP.DenseCondensedKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
opt = MadNLP.Options(; madnlp_options...)
ips = MadNLP.InteriorPointSolver{TKKTGPU}(lq_dense_KSEF, opt; option_linear_solver=copy(linear_solver_options))
sol_ref = MadNLP.optimize!(ips)

madnlp(lq_sparse_KSEF)