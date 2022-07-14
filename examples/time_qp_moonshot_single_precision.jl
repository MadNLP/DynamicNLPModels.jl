using Revise
using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU, CUDA, DelimitedFiles, SparseArrays, MKLSparse
using DataFrames, JLD, Printf, MadNLPHSL
include("build_thinplate.jl")

function convert_precision(lqdm, T; dense=true)
    dnlp = lqdm.dynamic_data

    N  = dnlp.N
    A  = T.(dnlp.A)
    B  = T.(dnlp.B)
    Q  = T.(dnlp.Q)
    R  = T.(dnlp.R)
    s0 = T.(dnlp.s0)

    E = T.(dnlp.E)
    F = T.(dnlp.F)
    if dnlp.K != nothing
        global K = T.(dnlp.K)
    else
        global K = nothing
    end
    Qf = T.(dnlp.Qf)
    S  = T.(dnlp.S)
    sl = T.(dnlp.sl)
    su = T.(dnlp.su)
    ul = T.(dnlp.ul)
    uu = T.(dnlp.uu)
    gl = T.(dnlp.gl)
    gu = T.(dnlp.gu)

    if dense
        return DenseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, K = K, Qf = Qf, S = S, sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    else
        return SparseLQDynamicModel(s0, A, B, Q, R, N; E = E, F = F, K = K, Qf = Qf, S = S, sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    end
end

function MadNLP.jac_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, jac) where {T, V, M1<: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_jac)

    J = nlp.data.A
    copyto!(jac, J)
end

function MadNLP.hess_dense!(nlp::DenseLQDynamicModel{T, V, M1, M2, M3}, x, w1l, hess; obj_weight = 1.0) where {T, V, M1<: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix}
    NLPModels.increment!(nlp, :neval_hess)
    H = nlp.data.H
    copyto!(hess, H)
end


function solve_CPU_sparse(lqdm, kkt_system)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>Ma27Solver,
            :print_level=>MadNLP.DEBUG,
            :jacobian_constant=>true,
            :hessian_constant=>true,
        )

        ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref = MadNLP.optimize!(ips)

        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end
    end
end

function time_CPU_sparse(lqdm_list, kkt_system)
    lens = length(lqdm_list)

    t       = zeros(lens)
    f       = zeros(lens)
    iters   = zeros(lens)
    status  = []
    ips_tt  = zeros(lens)
    ips_st  = zeros(lens)
    ips_eft = zeros(lens)
    ips_lst = zeros(lens)

    for i in 1:length(lqdm_list)

        lqdm = lqdm_list[i]

        iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_CPU_sparse(lqdm_list[i], kkt_system)
        push!(status, s)
    end

    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end

function solve_CPU_sparse32(lqdm, kkt_system)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>Ma27Solver,
            :print_level=>MadNLP.DEBUG,
            :jacobian_constant=>true,
            :hessian_constant=>true,
            :max_iter => 50,
        )

        ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref = MadNLP.optimize!(ips)

        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end
    end
end

function time_CPU_sparse32(lqdm_list, kkt_system)
    lens = length(lqdm_list)

    t       = zeros(lens)
    f       = zeros(lens)
    iters   = zeros(lens)
    status  = []
    ips_tt  = zeros(lens)
    ips_st  = zeros(lens)
    ips_eft = zeros(lens)
    ips_lst = zeros(lens)

    for i in 1:length(lqdm_list)

        lqdm = lqdm_list[i]

        iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_CPU_sparse32(lqdm_list[i], kkt_system)
        push!(status, s)
    end

    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end

function solve_CPU(lqdm, kkt_system, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>LapackCPUSolver,
            :jacobian_constant=>true,
            :hessian_constant=>true,
            :lapack_algorithm => algorithm
        )

        ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref      = MadNLP.optimize!(ips)
        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end

    end
end

function solve_CPU32(lqdm, kkt_system, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :linear_solver=>LapackCPUSolver,
            :jacobian_constant=>true,
            :hessian_constant=>true,
            :max_iter => 50,
            :tol => 1e-3,
            :acceptable_tol => 1e-4,
            :lapack_algorithm => algorithm
        )

        ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref      = MadNLP.optimize!(ips)
        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end

    end
end

function time_CPU(lqdm_list, kkt_system, algorithm; float32=false)
    lens = length(lqdm_list)

    t       = zeros(lens)
    f       = zeros(lens)
    iters   = zeros(lens)
    status  = []
    ips_tt  = zeros(lens)
    ips_st  = zeros(lens)
    ips_eft = zeros(lens)
    ips_lst = zeros(lens)

    for i in 1:length(lqdm_list)

        if float32
            iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i]= solve_CPU32(lqdm_list[i], kkt_system, algorithm)
            push!(status, s)
        else
            iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i]= solve_CPU(lqdm_list[i], kkt_system, algorithm)
            push!(status, s)
        end
    end

    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end


function solve_GPU(lqdm, kkt_system, kkt_type, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
        :kkt_system=>kkt_system,
        :linear_solver=>LapackGPUSolver,
        :jacobian_constant=>true,
        :hessian_constant=>true,
        :lapack_algorithm => algorithm
        )

        ips = MadNLPGPU.CuInteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref = MadNLP.optimize!(ips)

        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end

    end
end


function solve_GPU32(lqdm, kkt_system, kkt_type, algorithm)

    for i in 1:2
        madnlp_options = Dict{Symbol, Any}(
        :kkt_system=>kkt_system,
        :linear_solver=>LapackGPUSolver,
        :jacobian_constant=>true,
        :hessian_constant=>true,
        :max_iter => 50,
        :tol => 1e-3,
        :acceptable_tol => 1e-4,
        :lapack_algorithm => algorithm
        )

        ips = MadNLPGPU.CuInteriorPointSolver(lqdm, option_dict = madnlp_options)
        sol_ref = MadNLP.optimize!(ips)

        if i == 2
            return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
        end

    end
end

function time_GPU(lqdm_list, kkt_system, kkt_type, algorithm; float32=false)
    lens = length(lqdm_list)

    t       = zeros(lens)
    f       = zeros(lens)
    iters   = zeros(lens)
    status  = []
    ips_tt  = zeros(lens)
    ips_st  = zeros(lens)
    ips_eft = zeros(lens)
    ips_lst = zeros(lens)

    for i in 1:length(lqdm_list)

        lqdm = lqdm_list[i]

        if float32
            iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_GPU32(lqdm, kkt_system, kkt_type, algorithm)
            push!(status, s)
        else
            iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_GPU(lqdm, kkt_system, kkt_type, algorithm)
            push!(status, s)
        end

    end

    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end

function build_lqdm(N_range, ns_range, nu_range; dense=true, sl = -Inf, su = Inf, ul = -Inf, uu = Inf, nc)

    lqdm_list = Any[]
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

                ns = j
                N  = i
                nu = k

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

                if dense
                    lqdm = build_thinplate(j, k, i, .001, .01; d = d, Tbar = 400., dense = true, sl = sl, su = su, ul = ul, uu = uu, K = K, S = S, E = E, F = F, gl = gl, gu = gu)
                else
                    lqdm = build_thinplate(j, k, i, .001, .01; d = d, Tbar = 400., dense = false, sl = sl, su = su, ul = ul, uu = uu, K = K, S = S, E = E, F = F, gl = gl, gu = gu)
                end
                println("Done with ", i, j, k)

                push!(lqdm_list, lqdm)
            end
        end
    end
    return lqdm_list
end


function run_timers(N_range, ns_range, nu_range, file_name1, n_vals; sl = -Inf, su = Inf, ul = -Inf, uu = Inf, nc = nc)

    #ts, ks        = time_CPU(MadNLP.SPARSE_KKT_SYSTEM, MadNLPLapackCPU, MadNLPLapackCPU.BUNCHKAUFMAN, n_range, ns_range, nu_range, false)

    lqdm_list  = build_lqdm(N_range, ns_range, nu_range; sl = sl, su = su, ul = ul, uu = uu, nc = nc)
    lqdm_list32 = []
    #lqdm_list   = build_lqdm(N_range, ns_range, nu_range; sl = sl, su = su, ul = ul, uu = uu, nc = nc, dense = false)
    for i in lqdm_list
        @time lqdm32 = convert_precision(i, Float32; dense=true)
        push!(lqdm_list32, lqdm32)
    end

    stats = Dict{Symbol, DataFrame}()

#    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, efcpu, lscpu    = time_CPU_sparse(lqdm_list, MadNLP.SPARSE_KKT_SYSTEM)
#    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = efcpu, lin_sol_time = lscpu, n_vals = n_vals)
#    stats[:CPU_MA27] = df

#    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, efcpu, lscpu    = time_CPU_sparse32(lqdm_list32, MadNLP.SPARSE_KKT_SYSTEM)
#    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = efcpu, lin_sol_time = lscpu, n_vals = n_vals)
#    stats[:CPU_MA27_32] = df

    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, efcpu, lscpu = time_CPU(lqdm_list, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.CHOLESKY)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = efcpu, lin_sol_time = lscpu, n_vals = n_vals)
    stats[:CPU_CH_64] = df

    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, efcpu, lscpu = time_CPU(lqdm_list32, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.CHOLESKY; float32=true)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = efcpu, lin_sol_time = lscpu, n_vals = n_vals)
    stats[:CPU_CH_32] = df


    tgpu, kgpu, sgpu, fgpu, ttgpu, stgpu, efgpu, lsgpu = time_GPU(lqdm_list, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLP.CHOLESKY)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tgpu)], f = fgpu, t = tgpu, status = sgpu, iter = kgpu, tot_time = ttgpu, sol_time = stgpu, fun_eval_time = efgpu, lin_sol_time = lsgpu, n_vals = n_vals)
    stats[:GPU_CH_64] = df

    tgpu, kgpu, sgpu, fgpu, ttgpu, stgpu, efgpu, lsgpu = time_GPU(lqdm_list32, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.DenseCondensedKKTSystem, MadNLP.CHOLESKY; float32=true)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tgpu)], f = fgpu, t = tgpu, status = sgpu, iter = kgpu, tot_time = ttgpu, sol_time = stgpu, fun_eval_time = efgpu, lin_sol_time = lsgpu, n_vals = n_vals)
    stats[:GPU_CH_32] = df

    JLD.save(file_name1, "data", stats)
end


N_range = 50
nu_range = 10
ns_range = [100, 300, 500]

run_timers(N_range, ns_range, nu_range, "Float32Test.jld", ns_range; sl = 300., su = 500., ul = -140., uu = 140., nc = 10)
