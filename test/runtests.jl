using Test, DynamicNLPModels, MadNLP, Random, JuMP, LinearAlgebra
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
su = sl .+ 4
uu = ul .+ 10
s0 = sl .+ 2


# Add Qf matrix
Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

# Test with no bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0)
dnlp        = LQDynamicData(s0, A, B, Q, R, N)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6



# Test with lower bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
dnlp        = LQDynamicData(s0, A, B, Q, R, N;  sl = sl, ul=ul)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6



# Test with upper bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; su = su, uu = uu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6



# Test with upper and lower bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-6
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-6

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6




# Test with Qf matrix
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, Qf=Qf)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-6
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-6

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6



# Test with E and F matrix bounds
E  = rand(3, ns)
F  = rand(3, nu)
gl = fill(-5.0, 3)
gu = fill(15.0, 3)

model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6


# Test edge case where one state is unbounded, other(s) is bounded
su[1] = Inf
sl[1] = -Inf

model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-6

@test size(lq_condense.data.A, 1) == size(E, 1) * 3 + sum(su .!= Inf .|| sl .!= -Inf) * N