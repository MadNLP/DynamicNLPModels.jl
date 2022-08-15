using Revise
using LinearAlgebra, SparseArrays, DynamicNLPModels, MadNLP, Random
using NLPModels, QuadraticModels
using MatrixEquations



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


function build_oscillating_masses_AB(num_masses, nu, dt)

    if num_masses < 2 * nu
        error("Number of masses must be at least 2x nu")
    end

    k = 1
    m = 1

    #states are ordered as [x_0, v_0, x_1, v_1, x_2, v_2, ...]

    ns = num_masses * 2

    A = zeros(ns, ns)
    B = zeros(ns, nu)

    A[2, 2] = 1
    A[2, 1] = -2 * k * dt /m
    A[2, 3] = k * dt / m

    A[ns, ns] = 1
    A[ns, ns - 1] = -2 * k * dt / m
    A[ns, ns - 3] = k * dt / m

    # Set position links for masses 2 through num_masses - 1
    for i in 1:2:(ns - 1)
        A[i, i] = 1
        A[i, i + 1] = dt
    end
    # Set velocity links for masses 2 - num_masses
    for i in 4:2:(ns - 2)
        A[i, i] = 1
        A[i, i + 1] = k * dt / m
        A[i, i - 1] = -2 * k * dt /m
        A[i, i - 3] = k * dt / m
    end

    u_floor = floor(ns / 2 / nu)

    # index is the number of actuators added to the system to this point
    index = 1
    for i in 1:ns
        if index <= nu && i >= index * u_floor
            B[i, index] = 1# * .0001
            B[i + div(ns, 2), index] = -1 #* .0001
            index += 1
        end
    end

    return A, B

end

A, B = build_oscillating_masses_AB(6, 3, 1)

function build_oscillating_masses(num_masses, nu, dt, N; dense = true)
    k = 1
    A, B = build_oscillating_masses_AB(num_masses, nu, dt)
    #B = zeros(size(B))
    ns = size(A, 1)
    nu = size(B, 2)

    len = .6 * (num_masses + 1)

    Q  = 100.0 * Matrix(LinearAlgebra.I, ns, ns)
    Qf = 100.0 * Matrix(LinearAlgebra.I, ns, ns)
    for i in 2:2:ns
        Q[i, i] = 1
        Qf[i, i] = 1
    end
    R  = 1.0 * Matrix(LinearAlgebra.I, nu, nu) / 100

    s0 = zeros(ns)
    w  = zeros(ns)
    d = zeros(ns, (N + 1))
    ul = zeros(nu); fill!(ul, -10000)
    uu = zeros(nu); fill!(uu, 10000)
    sl = zeros(ns)
    su = zeros(ns)
    Random.seed!(10)

    for (j, i) in enumerate(1:2:ns)
        s0[i]     = j * .6 + rand(-.1:.0001:.1)
        d[i, :]  .= j * .6
        sl[i]     = j * .6 - .6
        sl[i + 1] = -5
        su[i + 1] = 5
        su[i]     = j * .6 + .6
    end

    w[ns] += len * dt * k

    Q_scale = 1
    R_scale = 1

    X, eig, F = ared(A, B, R_scale, Q_scale)
    K = - F
    #K = .001 * Matrix(LinearAlgebra.I, nu, ns)
    #K = -.001 * ones(nu, ns)
    if dense
        #lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; w = w, K = K) #relaxed problem
        lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; w = w, ul = ul, uu = uu, sl = sl, su = su, K = K)
    else
        lqdm = SparseLQDynamicModel(s0, sparse(A), sparse(B), sparse(Q), sparse(R), N; w = w, ul = ul, uu = uu, sl = sl, su = su)
    end

    block_Q = SparseArrays.sparse([],[],eltype(Q)[], ns * (N + 1), ns * (N + 1))

    for i in 1:N
        block_Q[(1 + (i - 1) * ns):(ns * i), (1 + (i - 1) * ns):(ns * i)] = Q
    end

    block_Q[(1 + ns * N):end, (1 + ns * N):end] = Qf

    Qd    = zeros(size(d, 1))
    Qdvec = zeros(length(d))
    dQd   = 0

    for i in 1:N
        LinearAlgebra.mul!(Qd, Q, d[:, i])
        Qdvec[(1 + ns * (i - 1)):ns * i] = Qd

        dQd += LinearAlgebra.dot(Qd, d[:, i])
    end

    LinearAlgebra.mul!(Qd, Qf, d[:, N + 1])
    Qdvec[(1 + ns * N):end] = Qd

    dQd += LinearAlgebra.dot(Qd, d[:, N + 1])


    # Add c and c0 that result from (x-d)^T Q (x-d) in the objective function
    if dense
        block_A = lqdm.blocks.A
        block_B = lqdm.blocks.B

        As0 = zeros(size(block_A, 1))
        LinearAlgebra.mul!(As0, block_A, s0)
        dQB = zeros(nu * N)
        dQB_sub_block = zeros(nu)

        for i in 1:N
            B_sub_block = block_B[(1 + ns * (i - 1)):ns * i, :]
            for j in N:-1:i
                Qd_sub_block = Qdvec[(1 + ns * j):(ns * (j + 1))]
                LinearAlgebra.mul!(dQB_sub_block, B_sub_block', Qd_sub_block)

                dQB[(1 + nu * (j - i)):nu * (j - i + 1)] .+= dQB_sub_block
            end
        end


        lqdm.data.c0 += dQd / 2
        lqdm.data.c0 += -LinearAlgebra.dot(Qdvec, As0)
        lqdm.data.c  += - dQB
    else
        uvec = zeros(nu * N)
        full_Qd = vcat(Qdvec, uvec)

        lqdm.data.c0 += dQd / 2
        lqdm.data.c  += - full_Qd
    end

    return lqdm
end

lqdm = build_oscillating_masses(6, 3, .05, 200; dense=true)
#lqdms = build_oscillating_masses(6, 3, .1, 100; dense=false)
#sol = madnlp(lqdms)
madnlp_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=>LapackCPUSolver,
    :jacobian_constant=>true,
    :hessian_constant=>true,
    :lapack_algorithm=>MadNLP.CHOLESKY,
    :max_iter=>100,
    :print_level=>MadNLP.DEBUG,
    #:inertia_correction_method=>MadNLP.INERTIA_FREE
)

ips1 = MadNLP.InteriorPointSolver(lqdm, option_dict = madnlp_options)
sol_ref1 = MadNLP.optimize!(ips1)

using MadNLPGPU
madnlp_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=>LapackGPUSolver,
    :jacobian_constant=>true,
    :hessian_constant=>true,
    :lapack_algorithm => MadNLP.CHOLESKY,
    :max_iter=>200
)

ips2 = MadNLPGPU.CuInteriorPointSolver(lqdm, option_dict = madnlp_options)
sol_ref2 = MadNLP.optimize!(ips2)
