using Test
include("JuMP_sparse_linear_mpc.jl")
include("QP_sparse_linear_mpc.jl")

nt = 3
ns = 2
nu = 1

Random.seed!(10)
Q_org = Random.rand(ns,ns)
Q = Q_org * transpose(Q_org) + I
R_org   = Random.rand(nu,nu)
R    = R_org * transpose(R_org) + I

A_org = rand(ns, ns)
A = A_org * transpose(A_org) + I
B = rand(ns, nu)


lvar = rand(0:.0000001:5, (ns*nt + nu*nt))
uvar = lvar .+ 10




mlb  = build_QP_JuMP_model(Q,R,A,B, nt; lvar = lvar)
mub  = build_QP_JuMP_model(Q,R,A,B, nt; uvar= uvar)
mulb = build_QP_JuMP_model(Q,R,A,B, nt; lvar = lvar, uvar= uvar)

qplb  = get_QM(Q, R, A, B, nt; lvar=lvar)
qpub  = get_QM(Q, R, A, B, nt; uvar=uvar)
qpulb = get_QM(Q, R, A, B, nt; lvar = lvar, uvar=uvar)

optimize!(mlb)
optimize!(mub)
optimize!(mulb)

ipopt_lb  = ipopt(qplb)
ipopt_ub  = ipopt(qpub)
ipopt_ulb = ipopt(qpulb)

@test abs(objective_value(mlb) - ipopt_lb.objective) < 1e-7
@test abs(objective_value(mub) - ipopt_ub.objective) < 1e-7
@test abs(objective_value(mulb) - ipopt_ulb.objective) < 1e-7

madnlp_lb  = madnlp(qplb, max_iter=100)
madnlp_ub  = madnlp(qpub, max_iter = 100)
madnlp_ulb = madnlp(qpulb, max_iter = 100)

