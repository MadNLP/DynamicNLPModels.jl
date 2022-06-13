using Revise

using DynamicNLPModels
using Random
using LinearAlgebra
using SparseArrays
N  = 3 # number of time steps
ns = 2 # number of states
nu = 1 # number of inputs

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

dnlp = LQDynamicData(s0, A, B, Q, R, N;)

LQDynamicModel(dnlp; condense=false)


dnlpe = LQDynamicData(ns, nu, N)

get_s0(dnlpe)
get_sl(dnlpe)
get_su(dnlpe)


set_Q!(dnlpe, 1,1,3.0)

dnlpe.Q