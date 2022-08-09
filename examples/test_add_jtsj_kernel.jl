using Revise
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays, CUDA
using Test

include("build_thinplate.jl")
include("add_jtsj_kernel.jl")

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

ns_vals = [10, 30, 50, 100, 500, 1000, 2000, 4000, 6000]
nu = 10
N  = 50
add_ns_cuda  = []
add_ns_cuda2 = []
add_ns_d     = []
add_ns_imp   = []

for i in ns_vals
    @time lqdm_d   = build_lqdm(i, nu, N; implicit=false)
    println("built full Jacobian lqdm")
    @time lqdm_imp = build_lqdm(i, nu, N; implicit=true)
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

    H     = zeros(nu * N, nu * N)
    H_imp = zeros(nu * N, nu * N)

    J     = get_jacobian(lqdm_d)
    J_imp = get_jacobian(lqdm_imp)
    J_cu   = CuArray{Float64}(J)
    J_cu_imp = get_jacobian(lqdm_CUDA)
    ΣJ    = similar(J); fill!(ΣJ, 0)
    ΣJcu = CuArray{Float64}(ΣJ)

    x     = rand(size(J, 1))

    xcuda = CuArray{Float64}(x)
    Hcuda1 = CuArray{Float64}(H_imp)
    Hcuda2 = CuArray{Float64}(H)
    println()
    println("The time splits are")
    a = @elapsed jtsj_mul!(H, J, x, ΣJ)
    #b = @elapsed add_JTSJ!_kernel(J_imp, x, H_imp)
    c = CUDA.@elapsed add_JTSJ!_kernel(J_cu_imp, xcuda, Hcuda1)
    d = CUDA.@elapsed jtsj_mul!(Hcuda2, J_cu, xcuda, ΣJcu)
    println()
    println(sum(abs.(LowerTriangular(H) - LowerTriangular(Array(Hcuda1)))))

    push!(add_ns_d, a)
    #push!(add_ns_imp, b)
    push!(add_ns_cuda, c)
    push!(add_ns_cuda2, d)
    #println(CUDA.memory_status())
    println(i)
    CUDA.reclaim()
end

println(ns_vals, "   ", add_ns_cuda, "   ", add_ns_cuda2, "   ", add_ns_d, "   ", add_ns_imp)
