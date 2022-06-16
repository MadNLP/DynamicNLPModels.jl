using Test, DynamicNLPModels, MadNLP, NLPModelsIpopt, Random, JuMP, Ipopt, LinearAlgebra
include("sparse_lq_test.jl")

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
su = sl .+ 20
uu = ul .+ 10
s0 = sl .+ 5


# Add Qf matrix
Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

# Test with no bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0)
dnlp        = LQDynamicData(s0, A, B, Q, R, N)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

optimize!(model)
madnlp_sol_ref_sparse   = madnlp(lq_sparse, max_iter=100) 
madnlp_sol_ref_condense = madnlp(lq_condense, max_iter=100)
ipopt_sol_ref_sparse    = ipopt(lq_sparse)
ipopt_sol_ref_condense  = ipopt(lq_condense)

@test objective_value(model) ≈ madnlp_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ madnlp_sol_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_condense.objective atol = 1e-7

@test madnlp_sol_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ madnlp_sol_ref_condense.solution atol =  1e-6



# Test with lower bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
dnlp        = LQDynamicData(s0, A, B, Q, R, N;  sl = sl, ul=ul)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

optimize!(model)
madnlp_sol_ref_sparse   = madnlp(lq_sparse, max_iter=100) 
madnlp_sol_ref_condense = madnlp(lq_condense, max_iter=100)
ipopt_sol_ref_sparse    = ipopt(lq_sparse)
ipopt_sol_ref_condense  = ipopt(lq_condense)

@test objective_value(model) ≈ madnlp_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ madnlp_sol_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_condense.objective atol = 1e-7

@test madnlp_sol_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ madnlp_sol_ref_condense.solution atol =  1e-6



# Test with upper bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; su = su, uu = uu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

optimize!(model)
madnlp_sol_ref_sparse   = madnlp(lq_sparse, max_iter=100) 
madnlp_sol_ref_condense = madnlp(lq_condense, max_iter=100)
ipopt_sol_ref_sparse    = ipopt(lq_sparse)
ipopt_sol_ref_condense  = ipopt(lq_condense)

@test objective_value(model) ≈ madnlp_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ madnlp_sol_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_condense.objective atol = 1e-7

@test madnlp_sol_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ madnlp_sol_ref_condense.solution atol =  1e-6



# Test with upper and lower bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

optimize!(model)
madnlp_sol_ref_sparse   = madnlp(lq_sparse, max_iter=100) 
madnlp_sol_ref_condense = madnlp(lq_condense, max_iter=100)
ipopt_sol_ref_sparse    = ipopt(lq_sparse)
ipopt_sol_ref_condense  = ipopt(lq_condense)

@test objective_value(model) ≈ madnlp_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ madnlp_sol_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_condense.objective atol = 1e-7

@test madnlp_sol_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ madnlp_sol_ref_condense.solution atol =  1e-6




# Test with Qf matrix
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, Qf=Qf)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

optimize!(model)
madnlp_sol_ref_sparse   = madnlp(lq_sparse, max_iter=100) 
madnlp_sol_ref_condense = madnlp(lq_condense, max_iter=100)
ipopt_sol_ref_sparse    = ipopt(lq_sparse)
ipopt_sol_ref_condense  = ipopt(lq_condense)

@test objective_value(model) ≈ madnlp_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ madnlp_sol_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ ipopt_sol_ref_condense.objective atol = 1e-7

@test madnlp_sol_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ madnlp_sol_ref_condense.solution atol =  1e-6
