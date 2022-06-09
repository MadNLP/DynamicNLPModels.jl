using Test
include("JuMP_sparse_linear_mpc.jl")
include("QP_sparse_linear_mpc.jl")

nt = 3 # number of time steps
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
lvar = rand(0:.0000001:1, (ns*nt + nu*nt))
uvar = lvar .+ 10


# build JuMP models
mnb  = build_QP_JuMP_model(Q,R,A,B, nt)
mlb  = build_QP_JuMP_model(Q,R,A,B, nt; lvar = lvar)
mub  = build_QP_JuMP_model(Q,R,A,B, nt; uvar= uvar)
mulb = build_QP_JuMP_model(Q,R,A,B, nt; lvar = lvar, uvar= uvar)

# Build Quadratic Model
qpnb  = get_QM(Q, R, A, B, nt)
qplb  = get_QM(Q, R, A, B, nt; lvar=lvar)
qpub  = get_QM(Q, R, A, B, nt; uvar=uvar)
qpulb = get_QM(Q, R, A, B, nt; lvar = lvar, uvar=uvar)


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

mulbf  = build_QP_JuMP_model(Q,R,A,B, nt; lvar = lvar, uvar= uvar, Qf = Qf)
qpulbf = get_QM(Q, R, A, B, nt; lvar = lvar, uvar=uvar, Qf = Qf)

# Solve new problem with Qf matrix
optimize!(mulbf)
ipopt_ulbf  = ipopt(qpulbf)
madnlp_ulbf = madnlp(qpulbf)

# Test results
@test abs(objective_value(mulbf) - ipopt_ulbf.objective)  < 1e-7
@test abs(objective_value(mulbf) - madnlp_ulbf.objective) < 1e-7


# Test edge cases of model functions 
nt = 50 # number of time steps
ns = 50 # number of states
nu = 10 # number of inputs

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
lvar = rand(0:.0000001:1, (ns*nt + nu*nt))
uvar = lvar .+ 10

Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * transpose(Qf_rand) + I

# Generate the big problems; these won't be solved because random bounds makes it likely infeasible
mbig  = build_QP_JuMP_model(Q,R,A,B, nt; lvar=lvar, uvar=uvar, Qf=Qf)
qpbig = get_QM(Q, R, A, B, nt; lvar=lvar, uvar=uvar, Qf=Qf)

# Test if matrices formed correctly
@test length(qpbig.data.H.rowval) == nt*ns^2 + (nt-1)*nu^2
@test qpbig.data.c == zeros(nt*ns + nu*nt)
@test qpbig.data.H[nt*ns + 1, nt*ns + 1] == R[1,1]
@test qpbig.data.H[(nt-1) * ns + 1, (nt-1)*ns + 1] == Qf[1,1]

