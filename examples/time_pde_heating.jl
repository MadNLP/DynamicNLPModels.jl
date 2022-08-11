using Revise
using DynamicNLPModels, NLPModels, Random, LinearAlgebra, MadNLP, QuadraticModels, MadNLPGPU, CUDA, DelimitedFiles, SparseArrays, MKLSparse
using DataFrames, JLD, Printf, MadNLPHSL

include("PDE_boundary_3d_heating.jl")

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

function time_PDE(lqdm_list, kkt_system, algorithm, device)
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
        iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_PDE(lqdm_list[i], kkt_system, algorithm, device)
        push!(status, s)
    end
    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end

function solve_PDE(lqdm, kkt_system, algorithm, device)

    for i in 1:2
        if device == 1
            madnlp_options = Dict{Symbol, Any}(
                :kkt_system=>kkt_system,
                :linear_solver=>Ma27Solver,
                :jacobian_constant=>true,
                :hessian_constant=>true,
            )

            ips = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
            sol_ref = MadNLP.optimize!(ips)

            if i == 2
                return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, ips.cnt.total_time, ips.cnt.solver_time, ips.cnt.eval_function_time, ips.cnt.linear_solver_time
            end

        elseif device == 2
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

        elseif device == 3
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
end

function build_lqdm(N_range, nx_range, lenx, dt, Tmax, Tstart; dense::Bool = true)
    lqdm_list = []
    for i in N_range
        for j in nx_range

            lqdm = build_3D_PDE(i, j, lenx / j, dt, Tmax, Tstart; dense = dense)
            push!(lqdm_list, lqdm)

            println("Done with N = $i and nx = $j")
        end
    end
    return lqdm_list
end

function run_timers(N_range, nx_range, file_name1, n_vals, lenx, dt, Tmax, Tstart)
    lqdm_list_dense  = build_lqdm(N_range, nx_range, lenx, dt, Tmax, Tstart; dense=true)
    lqdm_list_sparse = build_lqdm(N_range, nx_range, lenx, dt, Tmax, Tstart; dense=false)

    stats = Dict{Symbol, DataFrame}()


    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, eftcpu, lstcpu    = time_PDE(lqdm_list_sparse, MadNLP.SPARSE_KKT_SYSTEM, 0, 1)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = eftcpu, lin_sol_time = lstcpu, n_vals = n_vals)
    stats[:CPU_MA27] = df

    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, eftcpu, lstcpu    = time_PDE(lqdm_list, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.CHOLESKY, 2)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = eftcpu, lin_sol_time = lstcpu, n_vals = n_vals)
    stats[:CPU_CH] = df

    tgpu, kgpu, sgpu, fgpu, ttgpu, stgpu, eftgpu, lstgpu   = time_GPU(lqdm_list, MadNLP.DENSE_CONDENSED_KKT_SYSTEM, MadNLP.CHOLESKY, 3)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tgpu)], f = fgpu, t = tgpu, status = sgpu, iter = kgpu, tot_time = ttgpu, sol_time = stgpu, fun_eval_time = eftgpu, lin_sol_time = lstgpu, n_vals = n_vals)
    stats[:GPU_CH] = df

    JLD.save(file_name1, "data", stats)

end


N_range = 50
nx_range = [4, 6, 8, 10, 12, 15, 20]

run_timers(N_range, nx_range, "nx_range_50N_test1.jld", nx_range, .5, 5, 350., 300.)
