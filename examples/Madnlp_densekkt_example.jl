using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU, CUDA, SparseArrays
include("build_thinplate.jl")


function MadNLP.jac_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, jac) where {T, V, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
    NLPModels.increment!(nlp, :neval_jac)
    J = nlp.data.A
    copyto!(jac, J)
end

function MadNLP.hess_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, w1l, hess; obj_weight = 1.0) where {T, V, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
    NLPModels.increment!(nlp, :neval_hess)
    H = nlp.data.H
    copyto!(hess, H)
end

function convert_to_CUDA!(lqdm::DenseLQDynamicModel)
    H = CuArray{Float64}(undef, size(lqdm.data.H))
    J = CuArray{Float64}(undef, size(lqdm.data.A))

    LinearAlgebra.copyto!(H, lqdm.data.H)
    LinearAlgebra.copyto!(J, lqdm.data.A)

    lqdm.data.H = H
    lqdm.data.A = J
end

# Define attributes of 1 D heat transfer model
N  = 10
ns = 5
nu = 5

dfunc = (i,k)->100*sin(2*pi*(4*i/N-12*k/ns)) + 400

d = zeros(ns, N+1)

for i in 1:(N+1)
    for j in 1:ns
        d[j,i] = dfunc(i,j)
    end
end

dx = .1
dt = .1

# Build model for 1 D heat transfer
lq_dense  = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = true)
lq_sparse = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 400., dense = false)

# Solve the dense problem
dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=> MadNLPLapackCPU,
    :max_iter=> 200
)

d_ips = MadNLP.InteriorPointSolver(lq_dense, option_dict = dense_options)
sol_ref_dense = MadNLP.optimize!(d_ips)


# Solve the sparse problem
sparse_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.SPARSE_KKT_SYSTEM,
    :linear_solver=>MadNLPLapackCPU,
    :print_level=>MadNLP.DEBUG,
)

s_ips = MadNLP.InteriorPointSolver(lq_sparse, option_dict = sparse_options)
sol_ref_sparse = MadNLP.optimize!(s_ips)



# Solve the dense problem on the GPU
gpu_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>MadNLPLapackGPU,
        :print_level=>MadNLP.DEBUG,
        :jacobian_constant=>true,
        :hessian_constant=>true,
)

convert_to_CUDA!(lq_dense)

TKKTGPU = MadNLP.DenseKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
opt = MadNLP.Options(; gpu_options...)
gpu_ips = MadNLP.InteriorPointSolver{TKKTGPU}(lq_dense, opt; option_linear_solver=copy(gpu_options))
sol_ref_gpu = MadNLP.optimize!(gpu_ips)
