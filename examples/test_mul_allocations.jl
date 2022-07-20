using Revise
using Random, SparseArrays, LinearAlgebra, DynamicNLPModels
# mul! with 2 for loops

N  = 10 # number of time steps
ns = 10 # number of states
nu = 10 # number of inputs

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

Qf_rand = Random.rand(ns,ns)
Qf = Qf_rand * Qf_rand' + I

E  = rand(3, ns)
F  = rand(3, nu)
gl = fill(-5.0, 3)
gu = fill(15.0, 3)

S = rand(ns, nu)

K = rand(nu, ns)

dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)
lq_imp    = DenseLQDynamicModel(dnlp; implicit=true)


x = rand(size(lq_dense.data.A, 1))
y = rand(size(lq_dense.data.A, 2))

x_imp = copy(x)
y_imp = copy(y)

J     = lq_dense.data.A
J_imp = lq_imp.data.A

mul!(x, J, y)
@time mul!(x, J, y)
mul!(x_imp, J_imp, y_imp)
@time mul!(x_imp, J_imp, y_imp)
