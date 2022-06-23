using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU

function model_to_CUDA(lqdm::DenseLQDynamicModel)
    meta         = lqdm.meta
    data         = lqdm.data
    blocks       = lqdm.blocks
    dynamic_data = lqdm.dynamic_data

    dnlp   = dynamic_data_to_CUDA(dynamic_data)
    data   = data_to_CUDA(data)
    dense_blocks = dense_blocks_to_CUDA(blocks)
    meta   = meta_to_CUDA(meta) 

    DenseLQDynamicModel(
        meta,
        NLPModels.Counters(),
        data,
        dnlp,
        dense_blocks
    )
end

function dynamic_data_to_CUDA(dnlp::LQDynamicData)
    s0c = CuVector{Float64}(undef, length(dnlp.s0))
    Ac  = CuArray{Float64}(undef, size(dnlp.A))
    Bc  = CuArray{Float64}(undef, size(dnlp.B))
    Qc  = CuArray{Float64}(undef, size(dnlp.Q))
    Rc  = CuArray{Float64}(undef, size(dnlp.R))
    Kc  = CuArray{Float64}(undef, size(dnlp.K))
    Sc  = CuArray{Float64}(undef, size(dnlp.S))
    Ec  = CuArray{Float64}(undef, size(dnlp.E))
    Fc  = CuArray{Float64}(undef, size(dnlp.F))
    Qfc = CuArray{Float64}(undef, size(dnlp.Qf))
    glc = CuVector{Float64}(undef, length(dnlp.gl))
    guc = CuVector{Float64}(undef, length(dnlp.gu))
    ulc = CuVector{Float64}(undef, length(dnlp.ul))
    uuc = CuVector{Float64}(undef, length(dnlp.uu))
    slc = CuVector{Float64}(undef, length(dnlp.sl))
    suc = CuVector{Float64}(undef, length(dnlp.su))

    LinearAlgebra.copyto!(Ac, dnlp.A)
    LinearAlgebra.copyto!(Bc, dnlp.B)
    LinearAlgebra.copyto!(Qc, dnlp.Q)
    LinearAlgebra.copyto!(Rc, dnlp.R)
    LinearAlgebra.copyto!(s0c, dnlp.s0)
    LinearAlgebra.copyto!(Kc, dnlp.K)
    LinearAlgebra.copyto!(Sc, dnlp.S)
    LinearAlgebra.copyto!(Ec, dnlp.E)
    LinearAlgebra.copyto!(Fc, dnlp.F)
    LinearAlgebra.copyto!(Qfc, dnlp.Qf)
    LinearAlgebra.copyto!(glc, dnlp.gl)
    LinearAlgebra.copyto!(guc, dnlp.gu)
    LinearAlgebra.copyto!(ulc, dnlp.ul)
    LinearAlgebra.copyto!(uuc, dnlp.uu)
    LinearAlgebra.copyto!(slc, dnlp.sl)
    LinearAlgebra.copyto!(suc, dnlp.su)

    println(typeof(s0c))

    LQDynamicData(s0c, Ac, Bc, Qc, Rc, dnlp.N; Qf = Qfc, S = Sc,  
    E = Ec, F = Fc, K = Kc, sl = slc, su = suc, ul = ulc, uu = uuc, gl = glc, gu = guc
    )
end

function data_to_CUDA(data::QuadraticModels.QPData)
    c = CuVector{Float64}(undef, length(data.c))
    H = CuArray{Float64}(undef, size(data.H))
    J = CuArray{Float64}(undef, size(data.A))

    LinearAlgebra.copyto!(c, data.c)
    LinearAlgebra.copyto!(H, data.H)
    LinearAlgebra.copyto!(J, data.A)


    QuadraticModels.QPData(
        data.c0, 
        c,
        H,
        J
    )
end


function dense_blocks_to_CUDA(db::Block_Matrices)
    A  = CuArray{Float64}(undef, size(db.A))
    B  = CuArray{Float64}(undef, size(db.B))
    E  = CuArray{Float64}(undef, size(db.E))
    F  = CuArray{Float64}(undef, size(db.F))
    gl = CuArray{Float64}(undef, size(db.gl))
    gu = CuArray{Float64}(undef, size(db.gu))
    
    LinearAlgebra.copyto!(A, db.A)
    LinearAlgebra.copyto!(B, db.B)
    LinearAlgebra.copyto!(E, db.E)
    LinearAlgebra.copyto!(F, db.F)
    LinearAlgebra.copyto!(gl, db.gl)
    LinearAlgebra.copyto!(gu, db.gu)
    
    Block_Matrices(A, B, db.Q, db.R, db.S, db.K, E, F, gl, gu)
end
    
function meta_to_CUDA(meta::NLPModels.NLPModelMeta)
    x0   = CuVector{Float64}(undef, length(meta.x0))
    lvar = CuVector{Float64}(undef, length(meta.lvar))
    uvar = CuVector{Float64}(undef, length(meta.uvar))
    lcon = CuVector{Float64}(undef, length(meta.lcon))
    ucon = CuVector{Float64}(undef, length(meta.ucon))

    LinearAlgebra.copyto!(x0, meta.x0)
    LinearAlgebra.copyto!(lvar, meta.lvar)
    LinearAlgebra.copyto!(uvar, meta.uvar)
    LinearAlgebra.copyto!(lcon, meta.lcon)
    LinearAlgebra.copyto!(ucon, meta.ucon)

    NLPModels.NLPModelMeta(
        meta.nvar,
        x0   = x0,
        lvar = lvar, 
        uvar = uvar, 
        ncon = meta.ncon,
        lcon = lcon,
        ucon = ucon,
        nnzj = meta.nnzj,
        nnzh = meta.nnzh,
        lin  = 1:meta.ncon,
        islp = (meta.ncon == 0);
    )
end


function MadNLP.jac_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, jac) where {T, V, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
    NLPModels.increment!(nlp, :neval_jac)
    J = nlp.data.A
    
    jac .= J
end

function MadNLP.hess_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, w1l, hess; obj_weight = 1.0) where {T, V, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
    NLPModels.increment!(nlp, :neval_hess)
    H = nlp.data.H

    hess .= H
end


N  = 3 # number of time steps
ns = 2 # number of states
nu = 1 # number of inputs

# generate random Q, R, A, and B matrices
Random.seed!(10)
Q_rand = Random.rand(ns, ns)
Q = Q_rand * transpose(Q_rand) + I
R_rand   = Random.rand(nu,nu)
R    = R_rand * transpose(R_rand) + I

A_rand = rand(ns, ns)
A = A_rand * transpose(A_rand) + I
B = rand(ns, nu)

# generate upper and lower bounds
sl = rand(ns)
ul = fill(-15.0, nu)
su = sl .+ 4
uu = ul .+ 10
s0 = sl .+ 2

Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

E  = rand(3, ns)
F  = rand(3, nu)
gl = fill(-5.0, 3)
gu = fill(15.0, 3)

S = rand(ns, nu)

K = rand(nu, ns)




# Test with upper and lower bounds
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, K = K, S = S, E = E, F = F, gl = gl, gu = gu, Qf = Qf)
lq_sparse   = SparseLQDynamicModel(dnlp)
lq_dense = DenseLQDynamicModel(dnlp)


dense_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=> MadNLPLapackCPU,
    :max_iter=> 200
)

ipd = MadNLP.InteriorPointSolver(lq_dense, option_dict=dense_options)
sol_ref_dense = MadNLP.optimize!(ipd)


sparse_options = Dict{Symbol, Any}(
    :kkt_system => MadNLP.SPARSE_KKT_SYSTEM,
    :linear_solver=>MadNLPLapackCPU
)

ips = MadNLP.InteriorPointSolver(lq_sparse, option_dict=sparse_options)

sol_ref_sparse = MadNLP.optimize!(ips)

lqdm_CUDA = model_to_CUDA(lq_dense)

dense_options_cuda = Dict{Symbol, Any}(
    :kkt_system => MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=> MadNLPLapackGPU,
    :max_iter=> 200
)

ips_d = MadNLP.InteriorPointSolver(lqdm_CUDA, option_dict=dense_options_cuda)