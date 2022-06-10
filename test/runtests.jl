using Test, DynamicNLPModels, MadNLP, NLPModelsIpopt, LinearAlgebra, Random, JuMP, Ipopt
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
ul = rand(nu)
su = sl .+ 10
uu = ul .+ 10


# build JuMP models
mnb  = build_QP_JuMP_model(Q,R,A,B, N)
mlb  = build_QP_JuMP_model(Q,R,A,B, N; sl = sl, ul = ul)
mub  = build_QP_JuMP_model(Q,R,A,B, N; su = su, uu = uu)
mulb = build_QP_JuMP_model(Q,R,A,B, N; sl = sl, su = su, ul = ul, uu = uu)

# Build Quadratic Model
qpnb  = get_QM(Q, R, A, B, N)
qplb  = get_QM(Q, R, A, B, N; sl = sl, ul = ul)
qpub  = get_QM(Q, R, A, B, N; su = su, uu = uu)
qpulb = get_QM(Q, R, A, B, N; sl = sl, ul = ul, su = su, uu = uu)


# Solve JuMP model with Ipopt
optimize!(mnb)
optimize!(mlb)
optimize!(mub)
optimize!(mulb)

# Solve QP with Ipopt
ipopt_nb  = ipopt(qpnb)
ipopt_lb  = ipopt(qplb)
ipopt_ub  = ipopt(qpub)
ipopt_ulb = ipopt(qpulb)

# Solve QP with MadNLP
madnlp_nb  = madnlp(qpnb, max_iter=50)
madnlp_lb  = madnlp(qplb, max_iter=50)
madnlp_ub  = madnlp(qpub, max_iter = 50)
madnlp_ulb = madnlp(qpulb, max_iter = 50)


# Test results
@test abs(objective_value(mnb) - ipopt_nb.objective)   < 1e-7
@test abs(objective_value(mlb) - ipopt_lb.objective)   < 1e-7
@test abs(objective_value(mub) - ipopt_ub.objective)   < 1e-7
@test abs(objective_value(mulb) - ipopt_ulb.objective) < 1e-7

@test abs(objective_value(mnb) - madnlp_nb.objective)   < 1e-7
@test abs(objective_value(mlb) - madnlp_lb.objective)   < 1e-7
@test abs(objective_value(mub) - madnlp_ub.objective)   < 1e-7
@test abs(objective_value(mulb) - madnlp_ulb.objective) < 1e-7


# Add Qf matrix
Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

mulbf  = build_QP_JuMP_model(Q,R,A,B, N; sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)
qpulbf = get_QM(Q, R, A, B, N; sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)

# Solve new problem with Qf matrix
optimize!(mulbf)
ipopt_ulbf  = ipopt(qpulbf)
madnlp_ulbf = madnlp(qpulbf)

# Test results
@test abs(objective_value(mulbf) - ipopt_ulbf.objective)  < 1e-7
@test abs(objective_value(mulbf) - madnlp_ulbf.objective) < 1e-7


# Test new problem with x0
s0 = sl .+ .5

mulbx0  = build_QP_JuMP_model(Q,R,A,B, N; sl = sl, su = su, ul = ul, uu = uu, s0 = s0)
qpulbx0 = get_QM(Q, R, A, B, N; sl = sl, su = su, ul= ul, uu = uu, s0=s0)

# Solve new problem with x0
optimize!(mulbx0)
ipopt_ulbx0   = ipopt(qpulbx0)
madnlp_ulbx0  = madnlp(qpulbx0)

# Test results
@test abs(objective_value(mulbx0) - ipopt_ulbx0.objective)  < 1e-7
@test abs(objective_value(mulbx0) - madnlp_ulbx0.objective) < 1e-7
@test abs(madnlp_ulbx0.objective  - ipopt_ulbx0.objective) < 1e-7