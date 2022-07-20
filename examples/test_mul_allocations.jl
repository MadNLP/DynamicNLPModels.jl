using Revise
using Random, SparseArrays, LinearAlgebra, DynamicNLPModels, CUDA
# mul! with 2 for loops

function dynamic_data_to_CUDA(dnlp::LQDynamicData)
    s0c = CuVector{Float64}(undef, length(dnlp.s0))
    Ac  = CuArray{Float64}(undef, size(dnlp.A))
    Bc  = CuArray{Float64}(undef, size(dnlp.B))
    Qc  = CuArray{Float64}(undef, size(dnlp.Q))
    Rc  = CuArray{Float64}(undef, size(dnlp.R))
    Kc  = CuArray{Float64}(undef, size(dnlp.K))
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
    LinearAlgebra.copyto!(Kc, dnlp.K)
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


    LQDynamicData(s0c, Ac, Bc, Qc, Rc, dnlp.N; Qf = Qfc, S = Sc,
    E = Ec, F = Fc, K = Kc, sl = slc, su = suc, ul = ulc, uu = uuc, gl = glc, gu = guc
    )
end

N  = 100 # number of time steps
ns = 100 # number of states
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
lq_dense  = DenseLQDynamicModel(dnlp)
lq_imp    = DenseLQDynamicModel(dnlp; implicit=true)

dnlp_CUDA = dynamic_data_to_CUDA(dnlp)

lq_cuda  = DenseLQDynamicModel(dnlp_CUDA; implicit=true)


x = rand(size(lq_dense.data.A, 1))
y = rand(size(lq_dense.data.A, 2))

x_imp = copy(x)
y_imp = copy(y)


J      = lq_dense.data.A
J_imp  = lq_imp.data.A
J_cuda = lq_cuda.data.A

mul!(x, J, y)
@time mul!(x, J, y)
mul!(x_imp, J_imp, y_imp)
@time mul!(x_imp, J_imp, y_imp)

xcu = CuArray{Float64}(x)
ycu = CuArray{Float64}(y)

mul!(xcu, J_cuda, ycu)
@time mul!(xcu, J_cuda, ycu)

#=

nu = 10
N  = 50
ns_list = [10, 50, 100, 200, 300, 500, 800, 1000,1200, 1500, 2000, 2500, 3000]

mul_times_ns_adj = []
add_times_ns_adj = []
counter = 0
GC.enable(false)
for i in ns_list
    lqdm_d   = build_lqdm(i, nu, N; implicit=false)
    lqdm_imp = build_lqdm(i, nu, N; implicit=true)

    x = rand(size(lqdm_d.data.A, 1))
    y = rand(size(lqdm_d.data.A, 2))
    y_imp = copy(y)
    x_imp = copy(x)
    a = @elapsed mul!(y, lqdm_d.data.A', x)
    b = @elapsed mul!(y_imp, lqdm_imp.data.A', x)
    push!(mul_times_ns_adj, a)
    push!(add_times_ns_adj, b)
    counter += 1
    println(counter)
end
GC.enable(true)
=#
