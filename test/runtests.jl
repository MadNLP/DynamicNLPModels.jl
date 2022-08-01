using Test, DynamicNLPModels, MadNLP, Random, JuMP, LinearAlgebra, SparseArrays, CUDA
include("sparse_lq_test.jl")

function test_mul(lq_dense, lq_dense_imp; cuda=false)
    dnlp = lq_dense.dynamic_data
    N    = dnlp.N
    nu   = dnlp.nu

    J     = get_jacobian(lq_dense)
    J_imp = get_jacobian(lq_dense_imp)

    Random.seed!(10)
    x = rand(nu * N)
    y = rand(size(J, 1))

    if cuda
        x_imp = CuArray(x)
        y_imp = CuArray(y)
        copyto!(x_imp, x)
        copyto!(y_imp, y)
    else
        x_imp = copy(x)
        y_imp = copy(y)
    end

    LinearAlgebra.mul!(y, J, x)
    LinearAlgebra.mul!(y_imp, J_imp, x_imp)

    @test y ≈ Vector(y_imp) atol = 1e-14

    x = rand(nu * N)
    y = rand(size(J, 1))

    if cuda
        x_imp = CuArray(x)
        y_imp = CuArray(y)
        copyto!(x_imp, x)
        copyto!(y_imp, y)
    else
        x_imp = copy(x)
        y_imp = copy(y)
    end

    LinearAlgebra.mul!(x, J', y)
    LinearAlgebra.mul!(x_imp, J_imp', y_imp)

    @test x ≈ Vector(x_imp) atol = 1e-14
end

function test_add_jtsj(lq_dense, lq_dense_imp; cuda=false)
    dnlp = lq_dense.dynamic_data
    N    = dnlp.N
    nu   = dnlp.nu

    H     = zeros(nu * N, nu * N)

    Random.seed!(10)
    J     = get_jacobian(lq_dense)
    J_imp = get_jacobian(lq_dense_imp)
    ΣJ    = similar(J); fill!(ΣJ, 0)

    x     = rand(size(J, 1))

    if cuda
        H_imp = CuArray(H)
        x_imp = CuArray(x)
        copyto!(H_imp, H)
        copyto!(x_imp, x)
    else
        H_imp = zeros(nu * N, nu * N)
        x_imp = copy(x)
    end

    LinearAlgebra.mul!(ΣJ, Diagonal(x), J)
    LinearAlgebra.mul!(H, J', ΣJ)

    add_jtsj!(H_imp, J_imp, x_imp)

    @test LowerTriangular(Array(H_imp)) ≈ LowerTriangular(H) atol = 1e-10
end

function dynamic_data_to_CUDA(dnlp::LQDynamicData)
    s0c = CuVector{Float64}(undef, length(dnlp.s0))
    Ac  = CuArray{Float64}(undef, size(dnlp.A))
    Bc  = CuArray{Float64}(undef, size(dnlp.B))
    Qc  = CuArray{Float64}(undef, size(dnlp.Q))
    Rc  = CuArray{Float64}(undef, size(dnlp.R))
    Sc  = CuArray{Float64}(undef, size(dnlp.S))
    Ec  = CuArray{Float64}(undef, size(dnlp.E))
    Fc  = CuArray{Float64}(undef, size(dnlp.F))
    Qfc = CuArray{Float64}(undef, size(dnlp.Qf))
    glc = CuVector{Float64}(undef, length(dnlp.gl))
    guc = CuVector{Float64}(undef, length(dnlp.gu))
    ulc = CuVector{Float64}(undef, length(dnlp.ul))
    uuc = CuVector{Float64}(undef, length(dnlp.uu))
    slc = CuVector{Float64}(undef, length(dnlp.sl))
    suc = CuVector{Float64}(undef, length(dnlp.su))

    LinearAlgebra.copyto!(Ac, dnlp.A)
    LinearAlgebra.copyto!(Bc, dnlp.B)
    LinearAlgebra.copyto!(Qc, dnlp.Q)
    LinearAlgebra.copyto!(Rc, dnlp.R)
    LinearAlgebra.copyto!(s0c, dnlp.s0)
    LinearAlgebra.copyto!(Sc, dnlp.S)
    LinearAlgebra.copyto!(Ec, dnlp.E)
    LinearAlgebra.copyto!(Fc, dnlp.F)
    LinearAlgebra.copyto!(Qfc, dnlp.Qf)
    LinearAlgebra.copyto!(glc, dnlp.gl)
    LinearAlgebra.copyto!(guc, dnlp.gu)
    LinearAlgebra.copyto!(ulc, dnlp.ul)
    LinearAlgebra.copyto!(uuc, dnlp.uu)
    LinearAlgebra.copyto!(slc, dnlp.sl)
    LinearAlgebra.copyto!(suc, dnlp.su)

    if dnlp.K != nothing
        Kc  = CuArray{Float64}(undef, size(dnlp.K))
        LinearAlgebra.copyto!(Kc, dnlp.K)
    else
        Kc = nothing
    end

    LQDynamicData(s0c, Ac, Bc, Qc, Rc, dnlp.N; Qf = Qfc, S = Sc,
    E = Ec, F = Fc, K = Kc, sl = slc, su = suc, ul = ulc, uu = uuc, gl = glc, gu = guc
    )
end

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
dnlp      = LQDynamicData(s0, A, B, Q, R, N)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6

# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test with lower bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul)
dnlp      = LQDynamicData(s0, A, B, Q, R, N;  sl = sl, ul=ul)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test with upper bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, su = su, uu = uu)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; su = su, uu = uu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; su = su, uu = uu)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-7

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test with upper and lower bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl=sl, ul=ul, su = su, uu = uu)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-6
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-6

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

# Test with Qf matrix
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, Qf=Qf)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, Qf = Qf)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-6
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-6

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

# Test get_u and get_s functions with no K matrix
s_values = value.(all_variables(model)[1:(ns * (N + 1))])
u_values = value.(all_variables(model)[(1 + ns * (N + 1)):(ns * (N + 1) + nu * N)])


@test s_values ≈ get_s(solution_ref_sparse, lq_sparse) atol = 1e-7
@test u_values ≈ get_u(solution_ref_sparse, lq_sparse) atol = 1e-7
@test s_values ≈ get_s(solution_ref_dense, lq_dense) atol = 1e-6
@test u_values ≈ get_u(solution_ref_dense, lq_dense) atol = 1e-7


# Test with E and F matrix bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

# Test edge case where one state is unbounded, other(s) is bounded
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl_with_inf, ul = ul, su = su_with_inf, uu = uu, E = E, F = F, gl = gl, gu = gu)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6

@test size(lq_dense.data.A, 1) == size(E, 1) * 3 + sum(su_with_inf .!= Inf .|| sl_with_inf .!= -Inf) * N


# Test S matrix case
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, S = S)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test K matrix case without S
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test K matrix case with S
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

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
dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; sl = sl, ul = ul_with_inf, su = su, uu = uu_with_inf, E = E, F = F, gl = gl, gu = gu, K = K, S = S)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-5
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-5


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

# Test K with no bounds
model     = build_QP_JuMP_model(Q,R,A,B, N;s0=s0, E = E, F = F, gl = gl, gu = gu, K = K)
dnlp      = LQDynamicData(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)
lq_sparse = SparseLQDynamicModel(dnlp)
lq_dense  = DenseLQDynamicModel(dnlp)

lq_sparse_from_data = SparseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)
lq_dense_from_data  = DenseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, gl = gl, gu = gu, K = K)

optimize!(model)
solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
solution_ref_dense            = madnlp(lq_dense, max_iter=100)
solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

@test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
@test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
@test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

@test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-5
@test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-5


# Test mul! operators and LQJacobianOperator
lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)
test_mul(lq_dense, lq_dense_imp)
test_add_jtsj(lq_dense, lq_dense_imp)

# Test mul! operators with CUDA
if CUDA.has_cuda_gpu()
    dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
    lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)

    test_mul(lq_dense, lq_dense_cuda; cuda=true)
    test_add_jtsj(lq_dense, lq_dense_cuda; cuda=true)
end

# Test get_u and get_s functions with K matrix
s_values = value.(all_variables(model)[1:(ns * (N + 1))])
u_values = value.(all_variables(model)[(1 + ns * (N + 1)):(ns * (N + 1) + nu * N)])

@test s_values ≈ get_s(solution_ref_sparse, lq_sparse) atol = 1e-7
@test u_values ≈ get_u(solution_ref_sparse, lq_sparse) atol = 1e-7
@test s_values ≈ get_s(solution_ref_dense, lq_dense) atol = 1e-7
@test u_values ≈ get_u(solution_ref_dense, lq_dense) atol = 1e-7


# Test get_* and set_* functions

dnlp      = LQDynamicData(s0, A, B, Q, R, N; sl = sl, ul = ul, su = su, uu = uu, E = E, F = F, gl = gl, gu = gu, K = K, S = S)
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

@test length(get_jacobian(lq_dense_imp)) == length(lq_dense_imp.data.A)
@test size(get_jacobian(lq_dense_imp)) == size(lq_dense_imp.data.A)
@test isreal(get_jacobian(lq_dense_imp)) == isreal(lq_dense_imp.data.A)
@test eltype(get_jacobian(lq_dense_imp)) == eltype(lq_dense_imp.data.A)
