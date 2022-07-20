using Revise
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays, CUDA

include("build_thinplate.jl")

function jtsj_mul!(H, J, x, SJ)
    SJ .= x .* J

    LinearAlgebra.mul!(H, J', SJ, 1, 1)
end

function build_lqdm(ns, nu, N; implicit=false)

    d = zeros(ns, N+1)
    dfunc = (x,y)->100*sin(2*pi*(4*x/N-12*y/ns)) + 400
    for l in 1:(N+1)
        for m in 1:ns
            d[m,l] = dfunc(m,l)
        end
    end

    K = 0.5 * Matrix(I, nu, ns)
    S = -.001 * Matrix(I, ns, nu)
    E = zeros(ns - 1, ns)

    for i in 1:(ns - 1)
        E[i, i] = 1
        E[i, i + 1] = -1
    end

    F = zeros(ns - 1, nu)

    gl_val = -100. / (ns / nu)
    gu_val = 100. / (ns / nu)

    gl = fill(gl_val, ns - 1)
    gu = fill(gu_val, ns - 1)

    gl .*= rand(.8:.00001:1, ns - 1)
    gu .*= rand(.8:.00001:1, ns - 1)

    if implicit
        lqdm = build_thinplate(ns, nu, N, .001, .01; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu, implicit = implicit)
    else
        lqdm = build_thinplate(ns, nu, N, .001, .01; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu)
    end

    return lqdm
end

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
#ns_vals     = [10, 30, 50, 80, 100, 150, 300, 500, 800, 1200, 1600, 2500, 4000, 6000]
ns_vals = [10]#, 30, 50, 100, 300, 500, 1000, 2500, 4000, 6000]
mul_ns_cuda = []
mul_ns_d    = []
mul_ns_imp  = []
counter = 0
for i in ns_vals
    lqdm_d   = build_lqdm(i, 10, 50; implicit=false)
    println("built full Jacobian lqdm")
    lqdm_imp = build_lqdm(i, 10, 50; implicit=true)
    println("built implicit Jacobian lqdm")
    dnlpCUDA = dynamic_data_to_CUDA(lqdm_d.dynamic_data)
    println("converted dynamic data to CUDA")
    println()
    println(CUDA.memory_status())
    @time lqdm_CUDA = DenseLQDynamicModel(dnlpCUDA; implicit=true)
    println("built CUDA Jacobian")
    println(CUDA.memory_status())
    println()

    Random.seed!(10)
    x = rand(size(lqdm_d.data.A, 1))
    y = rand(size(lqdm_d.data.A, 2))
    y_imp = copy(y)
    x_imp = copy(x)

    ycuda = CuArray{Float64}(y)
    xcuda = CuArray{Float64}(x)

    J      = get_jacobian(lqdm_d)
    J_imp  = get_jacobian(lqdm_imp)

    #a = @elapsed mul!(x, J, y)
    #b = @elapsed mul!(x_imp, J_imp, y_imp)
    c = @elapsed mul!(xcuda, lqdm_CUDA.data.A, ycuda)
    push!(mul_ns_cuda, c)
    #push!(mul_ns_d, a)
    #push!(mul_ns_imp, b)
    #println(CUDA.memory_status())
    counter += 1
    println(counter, "  ", i)
    CUDA.reclaim()
end

using Plots, LaTeXStrings
plot(ns_vals, mul_ns_d, label="mul! (matrix_CPU)", xaxis=:log, yaxis=:log, legend=:topleft)
plot!(ns_vals, mul_ns_imp, label="mul! (LQJacOp_CPU)")
plot!(ns_vals, mul_ns_cuda, label="mul! (LQJacOP, gemm_batched_strided)")
xlabel!(L"Number of States ($N = 50, n_u = 10, n_c = n_s - 1$)")
ylabel!("Time (s)")

savefig("Jx_mul_comp.png")


#mul_ns_cuda  = [.0001711, .0001616, 0.0001494, 0.0001537, 0.0001589, 0.0001545, 0.0001554, 0.0001697, 0.0001551, 0.0001709, .0001673]
#ns_vals_cuda = [10, 30, 50, 80, 100, 150, 300, 500, 800, 1000, 1500]


#=
H     = zeros(nu * N, nu * N)
H_imp = zeros(nu * N, nu * N)

Random.seed!(10)
J     = get_jacobian(lqdm_d)
J_imp = get_jacobian(lqdm_imp)
ΣJ    = similar(J); fill!(ΣJ, 0)

x     = rand(size(J, 1))

LinearAlgebra.mul!(ΣJ, Diagonal(x), J)
LinearAlgebra.mul!(H, J', ΣJ)

add_jtsj!(H_imp, J_imp, x)
fill!(H_imp, 0.0)
@profile add_jtsj!(H_imp, J_imp, x)
pprof()
=#
