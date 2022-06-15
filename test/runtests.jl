using Test, DynamicNLPModels, MadNLP, NLPModelsIpopt, Random, JuMP, Ipopt, LinearAlgebra
include("sparse_lq_test.jl")

N  = 3 # number of time steps
ns = 2 # number of states
nu = 1 # number of inputs

# generate random Q, R, A, and B matrices
Random.seed!(10)
Q_rand = Random.rand(ns,ns)
Q = Q_rand * transpose(Q_rand) + I
R_rand   = Random.rand(nu,nu)
R    = R_rand * transpose(R_rand) + I

A_rand = rand(ns, ns)
A = A_rand * transpose(A_rand) + I
B = rand(ns, nu)

# generate upper and lower bounds
sl = rand(ns)
ul = ones(nu)*-15
su = sl .+ 20
uu = ul .+ 10
s0 = sl .+ 5


# build JuMP models
mnb  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0)
mlb  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
mub  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
mulb = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, su = su, ul = ul, uu = uu)

# Build LQDynamicData
dnlpnb  = LQDynamicData(s0, A, B, Q, R, N)
dnlplb  = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul)
dnlpub  = LQDynamicData(s0, A, B, Q, R, N; su = su, uu = uu)
dnlpulb = LQDynamicData(s0, A, B, Q, R, N; sl = sl, su = su, ul = ul, uu = uu)

# Build sparse format
lqpnb_sparse  = LQDynamicModel(dnlpnb; condense=false)
lqplb_sparse  = LQDynamicModel(dnlplb; condense=false)
lqpub_sparse  = LQDynamicModel(dnlpub; condense=false)
lqpulb_sparse = LQDynamicModel(dnlpulb; condense=false)

# Build condensed format
lqpnb_cond  = LQDynamicModel(dnlpnb; condense=true)
lqplb_cond  = LQDynamicModel(dnlplb; condense=true)
lqpub_cond  = LQDynamicModel(dnlpub; condense=true)
lqpulb_cond = LQDynamicModel(dnlpulb; condense=true)

# Solve JuMP model with Ipopt
optimize!(mnb)
optimize!(mlb)
optimize!(mub)
optimize!(mulb)

# Solve with Ipopt
ipopt_nbs  = ipopt(lqpnb_sparse)
ipopt_lbs  = ipopt(lqplb_sparse)
ipopt_ubs  = ipopt(lqpub_sparse)
ipopt_ulbs = ipopt(lqpulb_sparse)

# Solve with MadNLP
madnlp_nbs  = madnlp(lqpnb_sparse, max_iter=50)
madnlp_lbs  = madnlp(lqplb_sparse, max_iter=50)
madnlp_ubs  = madnlp(lqpub_sparse, max_iter = 50)
madnlp_ulbs = madnlp(lqpulb_sparse, max_iter = 50)

madnlp_nbc  = madnlp(lqpnb_cond, max_iter=50)
madnlp_lbc  = madnlp(lqplb_cond, max_iter=50)
madnlp_ubc  = madnlp(lqpub_cond, max_iter = 50)
madnlp_ulbc = madnlp(lqpulb_cond, max_iter = 50)



# Test results
@test abs(objective_value(mnb) - ipopt_nbs.objective)   < 1e-7
@test abs(objective_value(mlb) - ipopt_lbs.objective)   < 1e-7
@test abs(objective_value(mub) - ipopt_ubs.objective)   < 1e-7
@test abs(objective_value(mulb) - ipopt_ulbs.objective) < 1e-7

@test abs(objective_value(mnb) - madnlp_nbs.objective)   < 1e-7
@test abs(objective_value(mlb) - madnlp_lbs.objective)   < 1e-7
@test abs(objective_value(mub) - madnlp_ubs.objective)   < 1e-7
@test abs(objective_value(mulb) - madnlp_ulbs.objective) < 1e-7

@test abs(objective_value(mnb) - madnlp_nbc.objective)   < 1e-7
@test abs(objective_value(mlb) - madnlp_lbc.objective)   < 1e-7
@test abs(objective_value(mub) - madnlp_ubc.objective)   < 1e-7
@test abs(objective_value(mulb) - madnlp_ulbc.objective) < 1e-7

@test sum(abs.(madnlp_nbs.solution[(ns*(N+1)+1):(ns*(N+1) + nu*N)] .- madnlp_nbc.solution)) < 1e-6
@test sum(abs.(madnlp_lbs.solution[(ns*(N+1)+1):(ns*(N+1) + nu*N)] .- madnlp_lbc.solution)) < 1e-6
@test sum(abs.(madnlp_ubs.solution[(ns*(N+1)+1):(ns*(N+1) + nu*N)] .- madnlp_ubc.solution)) < 1e-6
@test sum(abs.(madnlp_ulbs.solution[(ns*(N+1)+1):(ns*(N+1) + nu*N)] .- madnlp_ulbc.solution)) < 1e-6



# Add Qf matrix
Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

mulbf    = build_QP_JuMP_model(Q,R,A,B, N;s0 = s0, sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)
dnlpulbf = LQDynamicData(s0, A, B, Q, R, N; sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)
lqpulbfs  = LQDynamicModel(dnlpulbf;condense=false)
lqpulbfc  = LQDynamicModel(dnlpulbf;condense=true)

# Solve new problem with Qf matrix
optimize!(mulbf)
ipopt_ulbfs  = ipopt(lqpulbfs)
madnlp_ulbfs = madnlp(lqpulbfs)
madnlp_ulbfc = madnlp(lqpulbfc)

# Test results
@test abs(objective_value(mulbf) - ipopt_ulbfs.objective)  < 1e-7
@test abs(objective_value(mulbf) - madnlp_ulbfs.objective) < 1e-7
@test abs(objective_value(mulbf) - madnlp_ulbfc.objective) < 1e-7

@test sum(abs.(madnlp_ulbfs.solution[(ns*(N+1)+1):(ns*(N+1) + nu*N)] .- madnlp_ulbfc.solution)) < 1e-6