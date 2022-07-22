using Revise
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays, CUDA

include("build_thinplate.jl")

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
#=
ns_vals = [10, 30, 50, 100, 300, 500, 1000, 2500, 4000, 6000]
mul_ns_cuda = []
mul_ns_d    = []
mul_ns_imp  = []
mul_ns_cuda2 = []
for i in ns_vals
    @time lqdm_d   = build_lqdm(i, 10, 50; implicit=false)
    println("built full Jacobian lqdm")
    @time lqdm_imp = build_lqdm(i, 10, 50; implicit=true)
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
    J_cu   = CuArray{Float64}(J)
    J_imp  = get_jacobian(lqdm_imp)

    a = @elapsed mul!(x, J, y)
    b = @elapsed mul!(x_imp, J_imp, y_imp)
    c = CUDA.@elapsed mul!(xcuda, lqdm_CUDA.data.A, ycuda)
    d = CUDA.@elapsed mul!(xcuda, J_cu, ycuda)
    push!(mul_ns_cuda, c)
    push!(mul_ns_d, a)
    push!(mul_ns_imp, b)
    push!(mul_ns_cuda2, d)
    #println(CUDA.memory_status())
    println(i)
    CUDA.reclaim()
end

println(ns_vals, "    ", mul_ns_cuda, "   ", mul_ns_d, "   ", mul_ns_imp, "   ", mul_ns_cuda2)
=#

#@profile mul!(xcuda, J_cu, ycuda)
#d = @elapsed mul!(xcuda, J_cu, ycuda)
#push!(mul_ns_cuda, c)
#push!(mul_ns_d, a)
#push!(mul_ns_imp, b)
#push!(mul_ns_cuda2, d)



ns_vals = [10, 30, 50, 80, 100, 300, 500 , 800, 2000, 4000, 6000]
mulT_ns_cuda = []
mulT_ns_d    = []
mulT_ns_imp  = []
mulT_ns_cuda2 = []
for i in ns_vals
    @time lqdm_d   = build_lqdm(i, 10, 50; implicit=false)
    println("built full Jacobian lqdm")
    @time lqdm_imp = build_lqdm(i, 10, 50; implicit=true)
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
    J_cu   = CuArray{Float64}(J)
    J_imp  = get_jacobian(lqdm_imp)
    J_imp_cu = get_jacobian(lqdm_CUDA)

    a = @elapsed mul!(y, J', x)
    b = @elapsed mul!(y_imp, J_imp', x_imp)
    c = CUDA.@elapsed mul!(ycuda, J_imp_cu', xcuda)
    d = CUDA.@elapsed mul!(ycuda, J_cu', xcuda)
    push!(mulT_ns_cuda, c)
    push!(mulT_ns_d, a)
    push!(mulT_ns_imp, b)
    push!(mulT_ns_cuda2, d)
    #println(CUDA.memory_status())
    println(i)
    CUDA.reclaim()
end

println(ns_vals, "    ", mulT_ns_cuda, "    ", mulT_ns_cuda2, "    ", mulT_ns_d, "   ", mulT_ns_imp)

#using Plots, LaTeXStrings
#plot(ns_vals[2:end], mulT_ns_d[2:end], label="mul! (matrix_CPU)", xaxis=:log, yaxis=:log, legend=:topleft)
#plot!(ns_vals[2:end], mulT_ns_imp[2:end], label="mul! (LQJacOp_CPU)")
#plot!(ns_vals[2:end], mulT_ns_cuda[2:end], label="mul! (LQJacOP, gemm_batched_strided)")
#plot!(ns_vals[2:end], mulT_ns_cuda2[2:end], label="mul!(Matrix_GPU)")
#xlabel!(L"Number of States ($N = 50, n_u = 10, n_c = n_s - 1$)")
#ylabel!("Time (s)")


#@time lqdm_d   = build_lqdm(500, 10, 50; implicit=false)
#println("built full Jacobian lqdm")
#@time lqdm_imp = build_lqdm(500, 10, 50; implicit=true)
#println("built implicit Jacobian lqdm")
#dnlpCUDA = dynamic_data_to_CUDA(lqdm_d.dynamic_data)
#println("converted dynamic data to CUDA")
#println()
#println(CUDA.memory_status())
#@time lqdm_CUDA = DenseLQDynamicModel(dnlpCUDA; implicit=true)
#println("built CUDA Jacobian")
#println(CUDA.memory_status())
#println()
#
#Random.seed!(10)
#x = rand(size(lqdm_d.data.A, 1))
#y = rand(size(lqdm_d.data.A, 2))
#y_imp = copy(y)
#x_imp = copy(x)
#
#ycuda = CuArray{Float64}(y)
#xcuda = CuArray{Float64}(x)
#
#J      = get_jacobian(lqdm_d)
#mul!(y, J', x)
#J_cu = get_jacobian(lqdm_CUDA)
#mul!(ycuda, J_cu', xcuda)
#CUDA.@time mul!(ycuda, J_cu', xcuda)
