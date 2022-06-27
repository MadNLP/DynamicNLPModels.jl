using Revise
using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU, CUDA
include("build_thinplate.jl")

CPUalgorithms = [MadNLPLapackCPU.BUNCHKAUFMAN, MadNLPLapackCPU.CHOLESKY, MadNLPLapackCPU.QR, MadNLPLapackCPU.LU]
GPUalgorithms = [MadNLPLapackGPU.BUNCHKAUFMAN, MadNLPLapackGPU.CHOLESKY, MadNLPLapackGPU.QR, MadNLPLapackGPU.LU]

function convert_to_CUDA!(lqdm::DenseLQDynamicModel)
    H = CuArray{Float64}(undef, size(lqdm.data.H))
    J = CuArray{Float64}(undef, size(lqdm.data.A))

    LinearAlgebra.copyto!(H, lqdm.data.H)
    LinearAlgebra.copyto!(J, lqdm.data.A)

    lqdm.data.H = H
    lqdm.data.A = J
end



function solve_CPU(lqdm, kkt_system, linear_solver, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>linear_solver,
            :print_level=>MadNLP.DEBUG,
        )

        ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options, lapackcpu_algorithm=algorithm)
        global sol_ref = MadNLP.optimize!(ips)
    end


    return sol_ref.iter, sol_ref.elapsed_time
end

function time_CPU(kkt_system, linear_solver, algorithm, N_range, ns_range, nu_range, dense = true)
    lens = length(N_range) * length(ns_range) * length(nu_range)

    t = zeros(lens)
    iters = zeros(lens)

    index = 1
    for i in N_range
        for j in ns_range
            for k in nu_range
                d = zeros(j, i+1)
                dfunc = (x,y)->100*sin(2*pi*(4*x/i-12*y/j)) + 400
                for l in 1:(i+1)
                    for m in 1:j
                        d[m,l] = dfunc(m,l)
                    end
                end

                if dense
                    lqdm = build_thinplate(j, k, i, .1, .1; d = d, Tbar = 400., dense = true)
                else
                    lqdm = build_thinplate(j, k, i, .1, .1; d = d, Tbar = 400., dense = false)
                end

                iters[index], t[index] = solve_CPU(lqdm, kkt_system, linear_solver, algorithm)
                index += 1

            end
        end
    end

    return t, iters
end

function solve_GPU(lqdm, kkt_system, kkt_type, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
        :kkt_system=>kkt_system,
        :linear_solver=>MadNLPLapackGPU,
        :print_level=>MadNLP.DEBUG,
        :jacobian_constant=>true,
        :hessian_constant=>true,
        )

        TKKTGPU = kkt_type{Float64, CuVector{Float64}, CuMatrix{Float64}}
        opt = MadNLP.Options(; madnlp_options...)
        ips = MadNLP.InteriorPointSolver{TKKTGPU}(lqdm, opt; option_linear_solver=copy(madnlp_options), lapackgpu_algorithm=algorithm)
        global sol_ref = MadNLP.optimize!(ips)
    end

    return sol_ref.iter, sol_ref.elapsed_time
end


function time_GPU(kkt_system, kkt_type, algorithm, N_range, ns_range, nu_range)
    lens = length(N_range) * length(ns_range) * length(nu_range)

    t = zeros(lens)
    iters = zeros(lens)

    index = 1
    for i in N_range
        for j in ns_range
            for k in nu_range
                d = zeros(j, i+1)
                dfunc = (x,y)->100*sin(2*pi*(4*x/i-12*y/j)) + 400
                for l in 1:(i+1)
                    for m in 1:j
                        d[m,l] = dfunc(m,l)
                    end
                end

                lqdm = build_thinplate(j, k, i, .1, .1; d = d, Tbar = 400., dense = true)
                convert_to_CUDA!(lqdm)

                iters[index], t[index] = solve_GPU(lqdm, kkt_system, kkt_type, algorithm)
                index += 1

            end
        end
    end

    return t, iters
end


function run_timers(n_range, ns_range, nu_range, file_name1, file_name2, headername_1, headername_2, n_vals)

    #ts, ks        = time_CPU(MadNLP.SPARSE_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, ns_range, nu_range, false)

    tcpu, kcpu    = time_CPU(MadNLP.DENSE_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, ns_range, nu_range,  true)

    t1 = hcat(n_vals, tcpu)
    k1 = kcpu


    tcpu, kcpu    = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, ns_range, nu_range, true)

    t1 = hcat(t1, tcpu)
    k1 = hcat(k1, kcpu)

    tcpu, kcpu    = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.CHOLESKY, n_range, ns_range, nu_range, true)

    t1 = hcat(t1, tcpu)
    k1 = hcat(k1, kcpu)

    tcpu, kcpu    = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.LU, n_range, ns_range, nu_range, true)

    t1 = hcat(t1, tcpu)
    k1 = hcat(k1, kcpu)

    tcpu, kcpu    = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.QR, n_range, ns_range, nu_range, true)

    t1 = hcat(t1, tcpu)
    k1 = hcat(k1, kcpu)


    tgpu, kgpu    = time_GPU(MadNLP.DENSE_KKT_SYSTEM, MadNLP.DenseKKTSystem, MadNLPLapackGPU.BUNCHKAUFMAN, n_range, ns_range, nu_range)

    t1 = hcat(t1, tgpu)
    k1 = hcat(k1, kgpu)

    tgpu, kgpu    = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.BUNCHKAUFMAN, n_range, ns_range, nu_range)

    t1 = hcat(t1, tgpu)
    k1 = hcat(k1, kgpu)

    tgpu, kgpu  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.CHOLESKY, n_range, ns_range, nu_range)

    t1 = hcat(t1, tgpu)
    k1 = hcat(k1, kgpu)

    tgpu, kgpu  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.LU, n_range, ns_range, nu_range)

    t1 = hcat(t1, tgpu)
    k1 = hcat(k1, kgpu)

    tgpu, kgpu  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.QR, n_range, ns_range, nu_range)

    t1 = hcat(t1, tgpu)
    k1 = hcat(k1, kgpu)


    
    open(file_name1; write=true) do f
        write(f, headername_1)
        writedlm(f, t1, ',')
    end


    open(file_name2; write=true) do f
        write(f, headername_2)
        writedlm(f, k1, ',')
    end

end

#file_name1 = "nu_range.csv"
#file_name2 = "nu_range_GPU.csv"
#headername_1 = "# nu values, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR,\n"
#headername_2 = "# nu values, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR,\n"
#
#N_range = 100
#nu_range = [10, 20, 30, 50]
#ns_range = 300
#
#run_timers(N_range, ns_range, nu_range, file_name1, file_name2, headername_1, headername_2)


file_name1 = "t_N_range_10nu_100ns_CPU.csv"
file_name2 = "k_N_range_10nu_100ns_GPU.csv"
headername_1 = "# N values, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR,\n"
headername_2 = "# N values, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR, dense, dense_condensed_BK, dense_condensed_CH, dense_condensed_LU, densed_condensed_QR,\n"

N_range = [10, 20, 50, 100, 300]
nu_range = 10
ns_range = 100

run_timers(N_range, ns_range, nu_range, file_name1, file_name2, headername_1, headername_2, N_range)



#=



n_range = [5, 10, 30, 50, 100, 200, 300, 400, 500]


ts, ks        = time_CPU(MadNLP.SPARSE_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, 10, 2, false)

td, kd        = time_CPU(MadNLP.DENSE_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, 10, 2, true)

tdc, kdc      = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, 10, 2, true)

tdcc, kdcc    = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.CHOLESKY, n_range, 10, 2, true)

tdclu, kdclu  = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.LU, n_range, 10, 2, true)

tdcqr, kdcqr  = time_CPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.QR, n_range, 10, 2, true)

tdg, kdg      = time_GPU(MadNLP.DENSE_KKT_SYSTEM, MadNLP.DenseKKTSystem, MadNLPLapackGPU.BUNCHKAUFMAN, n_range, 10, 2)

tdgc, kdgc    = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.BUNCHKAUFMAN, n_range, 10, 2)

tdgcc, kdgcc  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.CHOLESKY, n_range, 10, 2)

tdglu, kdglu  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.LU, n_range, 10, 2)

tdgqr, kdgqr  = time_GPU(MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLPLapackGPU.QR, n_range, 10, 2)



t = hcat(n_range, ts)
t = hcat(t, td)
t = hcat(t, tdc)
t = hcat(t, tdcc)
t = hcat(t, tdclu)
t = hcat(t, tdcqr)

tg = hcat(n_range, tdg)
tg = hcat(tg, tdgc)
tg = hcat(tg, tdgcc)
tg = hcat(tg, tdglu)
tg = hcat(tg, tdgqr)


using DelimitedFiles





plot(t[:,1], t[:,2])
plot(t[:,1], t[:,3])
plot!(t[:,1], t[:,4])
plot!(t[:,1], t[:,5])
plot!(t[:,1], t[:,6])
plot!(t[:,1], t[:,7])

=#