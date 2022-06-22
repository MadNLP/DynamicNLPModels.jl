using Test, DynamicNLPModels, MadNLP, Random, JuMP, LinearAlgebra
include("sparse_lq_test.jl")

N  = 3 # number of time steps
ns = 2 # number of states
nu = 1 # number of inputs

# generate random Q, R, A, and B matrices
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

su_with_inf = copy(su)
sl_with_inf = copy(sl)

su_with_inf[1] = Inf
sl_with_inf[1] = -Inf


Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * Qf_rand' + I

E  = rand(3, ns)
F  = rand(3, nu)
gl = fill(-5.0, 3)
gu = fill(15.0, 3)

S = rand(ns, nu)

K = rand(nu, ns)


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


# Test get_u and get_s functions with no K matrix
s_values = value.(all_variables(model)[1:(ns * (N + 1))])
u_values = value.(all_variables(model)[(1 + ns * (N + 1)):(ns * (N + 1) + nu * N)])


@test s_values ≈ get_s(solution_ref_sparse, lq_sparse) atol = 1e-7
@test u_values ≈ get_u(solution_ref_sparse, lq_sparse) atol = 1e-7
@test s_values ≈ get_s(solution_ref_condense, lq_condense) atol = 1e-6
@test u_values ≈ get_u(solution_ref_condense, lq_condense) atol = 1e-7


# Test with E and F matrix bounds
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
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu, condense=true)

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

@test size(lq_condense.data.A, 1) == size(E, 1) * 3 + sum(su_with_inf .!= Inf .|| sl_with_inf .!= -Inf) * N


# Test S matrix case
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S, condense=true)

optimize!(model)

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


# Test K matrix case without S
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, condense=true)

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



# Test K matrix case with S
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S, condense=true)

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


# Test K matrix case with S and with partial bounds on u

nu = 2 # number of inputs

# generate random Q, R, A, and B matrices
Random.seed!(3)
R_rand   = Random.rand(nu,nu)
R    = R_rand * transpose(R_rand) + I

B = rand(ns, nu)

# generate upper and lower bounds
ul = fill(-20.0, nu)
uu = ul .+ 30

ul_with_inf = copy(ul)
uu_with_inf = copy(uu)

uu_with_inf[1] = Inf
ul_with_inf[1] = -Inf

F  = rand(3, nu)

S = rand(ns, nu)

K = rand(nu, ns)


model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-5
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-5




# Test K with no bounds
model       = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp        = LQDynamicData(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

lq_sparse_from_data   = LQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K, condense=false)
lq_condense_from_data = LQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K, condense=true)

optimize!(model)
solution_ref_sparse             = madnlp(lq_sparse, max_iter=100) 
solution_ref_condense           = madnlp(lq_condense, max_iter=100)
solution_ref_sparse_from_data   = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_condense_from_data = madnlp(lq_condense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_condense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense.solution atol =  1e-5
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_condense_from_data.solution atol =  1e-5



# Test get_u and get_s functions with K matrix
s_values = value.(all_variables(model)[1:(ns * (N + 1))])
u_values = value.(all_variables(model)[(1 + ns * (N + 1)):(ns * (N + 1) + nu * N)])

@test s_values ≈ get_s(solution_ref_sparse, lq_sparse) atol = 1e-7
@test u_values ≈ get_u(solution_ref_sparse, lq_sparse) atol = 1e-7
@test s_values ≈ get_s(solution_ref_condense, lq_condense) atol = 1e-7
@test u_values ≈ get_u(solution_ref_condense, lq_condense) atol = 1e-7



# Test get_* and set_* functions

dnlp        = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse   = LQDynamicModel(dnlp; condense=false)
lq_condense = LQDynamicModel(dnlp; condense=true)

@test get_A(dnlp) == A
@test get_A(lq_sparse) == A
@test get_A(lq_condense) == A

rand_val = rand()
Qtest = copy(Q)

Qtest[1, 2] = rand_val
set_Q!(dnlp, 1,2, rand_val)
@test get_Q(dnlp) == Qtest

Qtest[1, 1] = rand_val
set_Q!(lq_sparse, 1, 1, rand_val)
@test get_Q(lq_sparse) == Qtest

Qtest[2, 1] = rand_val
set_Q!(lq_condense, 2, 1, rand_val)
@test get_Q(lq_condense) == Qtest


rand_val = rand()
gltest = copy(gl)

gltest[1] = rand_val
set_gl!(dnlp, 1, rand_val)
@test get_gl(dnlp) == gltest

gltest[2] = rand_val 
set_gl!(lq_sparse, 2, rand_val)
@test get_gl(lq_sparse) == gltest

gltest[3] = rand_val
set_gl!(lq_condense, 3, rand_val)
@test get_gl(lq_condense) == gltest
