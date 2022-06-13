using Revise

using DynamicNLPModels
using Random
using LinearAlgebra
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

LQDynamicData(s0, A, B, Q, R, N)