using Test, DynamicNLPModels, MadNLP, Random, JuMP, LinearAlgebra, SparseArrays, CUDA
include("sparse_lq_test.jl")
include("functions.jl")

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
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test with lower bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N;  sl = sl, ul=ul)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test with upper bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; su = su, uu = uu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test with upper and lower bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test with Qf matrix
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, Qf=Qf)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test with E and F matrix bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test edge case where one state is unbounded, other(s) is bounded
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

@test size(lq_dense.data.A, 1) == size(E, 1) * 3 + sum(su_with_inf .!= Inf .|| sl_with_inf .!= -Inf) * N


# Test S matrix case
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test K matrix case without S
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test K matrix case with S
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

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

model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test K with no bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)

runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)

# Test get_* and set_* functions

dnlp      = LQDynamicData(copy(s0), A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

@test get_A(dnlp) == A
@test get_A(lq_sparse) == A
@test get_A(lq_dense) == A

rand_val = rand()
Qtest = copy(Q)

Qtest[1, 2] = rand_val
set_Q!(dnlp, 1,2, rand_val)
@test get_Q(dnlp) == Qtest

Qtest[1, 1] = rand_val
set_Q!(lq_sparse, 1, 1, rand_val)
@test get_Q(lq_sparse) == Qtest

Qtest[2, 1] = rand_val
set_Q!(lq_dense, 2, 1, rand_val)
@test get_Q(lq_dense) == Qtest

rand_val = rand()
gltest = copy(gl)

gltest[1] = rand_val
set_gl!(dnlp, 1, rand_val)
@test get_gl(dnlp) == gltest

gltest[2] = rand_val
set_gl!(lq_sparse, 2, rand_val)
@test get_gl(lq_sparse) == gltest

gltest[3] = rand_val
set_gl!(lq_dense, 3, rand_val)
@test get_gl(lq_dense) == gltest


# Test non-default vector/matrix on GenericArrays
s0 = randn(Float32,2)
A  = randn(Float32,2,2)
B  = randn(Float32,2,2)
Q  = randn(Float32,2,2)
R  = randn(Float32,2,2)
S  = randn(Float32,2,2)
K  = randn(Float32,2,2)
E  = randn(Float32,2,2)
F  = randn(Float32,2,2)
gl = randn(Float32,2)
gu = gl .+ 2
sl = s0 .- 1
su = s0 .+ 1
ul = randn(Float32,2)
uu = ul .+ 2

s0 = Test.GenericArray(s0)
A  = Test.GenericArray(A)
B  = Test.GenericArray(B)
Q  = Test.GenericArray(Q)
R  = Test.GenericArray(R)
S  = Test.GenericArray(S)
K  = Test.GenericArray(K)
E  = Test.GenericArray(E)
F  = Test.GenericArray(F)
gl = Test.GenericArray(gl)
gu = Test.GenericArray(gu)
sl = Test.GenericArray(sl)
su = Test.GenericArray(su)
ul = Test.GenericArray(ul)
uu = Test.GenericArray(uu)

@test (DenseLQDynamicModel(s0, A, B, Q, R, 10; S = S, E = E, F = F, gl = gl, gu = gu, ul = ul, uu = uu, sl = sl, su = su) isa
    DenseLQDynamicModel{Float32, GenericArray{Float32, 1}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, Nothing})
@test (DenseLQDynamicModel(s0, A, B, Q, R, 10; K = K, S = S, E = E, F = F, gl = gl, gu = gu, ul = ul, uu = uu, sl = sl, su = su) isa
DenseLQDynamicModel{Float32, GenericArray{Float32, 1}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, GenericArray{Float32, 2}, GenericArray{Float32, 2}})
@test (SparseLQDynamicModel(s0, A, B, Q, R, 10; S = S, E = E, F = F, gl = gl, gu = gu, ul = ul, uu = uu, sl = sl, su = su) isa
    SparseLQDynamicModel{Float32, GenericArray{Float32, 1}, SparseMatrixCSC{Float32, Int64}, SparseMatrixCSC{Float32, Int64}, GenericArray{Float32, 2}, Nothing})
@test (SparseLQDynamicModel(s0, A, B, Q, R, 10; K = K, S = S, E = E, F = F, gl = gl, gu = gu, ul = ul, uu = uu, sl = sl, su = su) isa
    SparseLQDynamicModel{Float32, GenericArray{Float32, 1}, SparseMatrixCSC{Float32, Int64}, SparseMatrixCSC{Float32, Int64}, GenericArray{Float32, 2}, GenericArray{Float32, 2}})

# Test LQJacobianOperator APIs
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit=true)

@test length(get_jacobian(lq_dense_imp)) == (length(get_jacobian(lq_dense_imp).truncated_jac1)
    + length(get_jacobian(lq_dense_imp).truncated_jac2) + length(get_jacobian(lq_dense_imp).truncated_jac3))
(@test size(get_jacobian(lq_dense_imp))   == (size(get_jacobian(lq_dense_imp).truncated_jac1, 1) + size(get_jacobian(lq_dense_imp).truncated_jac2, 1)
    + size(get_jacobian(lq_dense_imp).truncated_jac3, 1), size(get_jacobian(lq_dense_imp).truncated_jac1, 2)))
@test isreal(get_jacobian(lq_dense_imp)) == isreal(get_jacobian(lq_dense_imp).truncated_jac1)
@test eltype(get_jacobian(lq_dense_imp)) == eltype(get_jacobian(lq_dense_imp).truncated_jac1)
