using Revise
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
ul = rand(nu)
su = sl .+ 10
uu = ul .+ 10
s0 = sl .+ 1


# build JuMP models
mnb  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0)
mlb  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
mub  = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
mulb = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, su = su, ul = ul, uu = uu)

# Build Quadratic Model
dnlpnb  = LQDynamicData(s0, A, B, Q, R, N)
dnlplb  = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul)
dnlpub  = LQDynamicData(s0, A, B, Q, R, N; su = su, uu = uu)
dnlpulb = LQDynamicData(s0, A, B, Q, R, N; sl = sl, su = su, ul = ul, uu = uu)

lqpnb  = LQDynamicModel(dnlpnb)
lqplb  = LQDynamicModel(dnlplb)
lqpub  = LQDynamicModel(dnlpub)
lqpulb = LQDynamicModel(dnlpulb)


# Solve JuMP model with Ipopt
optimize!(mnb)
optimize!(mlb)
optimize!(mub)
optimize!(mulb)

# Solve QP with Ipopt
ipopt_nb  = ipopt(lqpnb)
ipopt_lb  = ipopt(lqplb)
ipopt_ub  = ipopt(lqpub)
ipopt_ulb = ipopt(lqpulb)

# Solve QP with MadNLP
madnlp_nb  = madnlp(lqpnb, max_iter=50)
madnlp_lb  = madnlp(lqplb, max_iter=50)
madnlp_ub  = madnlp(lqpub, max_iter = 50)
madnlp_ulb = madnlp(lqpulb, max_iter = 50)


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

mulbf    = build_QP_JuMP_model(Q,R,A,B, N;s0 = s0, sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)
dnlpulbf = LQDynamicData(s0, A, B, Q, R, N; sl = sl, su = su, ul = ul, uu = uu, Qf = Qf)
lqpulbf  = LQDynamicModel(dnlpulbf)

# Solve new problem with Qf matrix
optimize!(mulbf)
ipopt_ulbf  = ipopt(lqpulbf)
madnlp_ulbf = madnlp(lqpulbf)

# Test results
@test abs(objective_value(mulbf) - ipopt_ulbf.objective)  < 1e-7
@test abs(objective_value(mulbf) - madnlp_ulbf.objective) < 1e-7