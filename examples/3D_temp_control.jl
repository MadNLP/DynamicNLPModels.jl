using Revise
using LinearAlgebra, SparseArrays, DynamicNLPModels, MadNLP, Random
using MatrixEquations, MadNLPGPU
using NLPModels, QuadraticModels


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
function build_3D_heating_AB(dx, nx, dt)

    A = zeros(nx^3, nx^3)
    B = zeros(nx^3, 6)

    k = 400. # thermal conductivity of copper, W/(m-K)
    k2 = 400
    rho = 8960. # density of copper, kg/m^3
    specificHeat = 386. # specific heat of copper, J/(kg-K)

    conduction_constant = k * dt / rho / specificHeat / dx^2
    input_constant = k2 * dt / rho / specificHeat / dx^2

    # Set A matrix
    for i in 1:nx^3
        A[i, i] = 1 - 6 * conduction_constant
        # Set links in x direction
        if i%nx != 0 && i%nx != 1
            A[i, i - 1] = conduction_constant
            A[i, i + 1] = conduction_constant
            #y has boundaries if i%100 < 10 or i %100 >90
            #z has boundaries if i%1000 < 100 and i%1000 > 900
            #A[i, i] += -2 * conduction_constant
        elseif i%nx == 0
            A[i, i - 1] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        else
            A[i, i + 1] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        end

        # Set links in the y direction
        if i%(nx^2) in 1:nx
            A[i, i + nx] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        elseif i%(nx^2) == 0 || i%(nx^2) > nx^2 - nx
            A[i, i - nx] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        else
            A[i, i + nx] = conduction_constant
            A[i, i - nx] = conduction_constant
            #A[i, i] += -2 * conduction_constant
        end

        # Set links in the z direction
        if i <= nx^2
            A[i, i + nx^2] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        elseif i > nx^3 - nx^2
            A[i, i - nx^2] = conduction_constant
            #A[i, i] += -1 * conduction_constant
        else
            A[i, i + nx^2] = conduction_constant
            A[i, i - nx^2] = conduction_constant
            #A[i, i] += -2 * conduction_constant
        end
    end

    #Set B matrix
    B[1:nx^2, 1] .= input_constant
    B[(nx^3 - nx^2):(nx^3), 2] .= input_constant
    for i in 1:nx^3
        if i%nx == 1
            B[i, 3] += input_constant
        end
        if i %nx == 0
            B[i, 4] += input_constant
        end
        if i%nx^2 in 1:nx
            B[i, 5] += input_constant
        end
        if i%nx^2 == 0 || i%nx^2 > nx^2 - nx
            B[i, 6] += input_constant
        end
    end
    return A, B
end

function set_d!(d, nx, N, Tmax, Tstart)
    fill!(d, Tstart)
    Tdiff = (Tmax - Tstart)/2
    Tmin = (Tmax - Tstart)/10
    for j in 1:(N + 1)
        for i in 1:nx^3
            x = i%nx
            y = div(i % nx^2, nx)
            z = div(i, nx^2)
            #if (x <= j || nx - x <= j) && (y <= j || nx - y <= j)
                #if z/nx <= .5
                #    d[i, j] = Tmax - z/nx * (2 * sin(3.14159 * x/nx) + 2 * sin(3.14159 * y/nx)) * Tdiff * ((N + 1 - j) / N / 3)
                #else
                    d[i, j] = Tstart + Tmin +  (1 - z/nx) * (2 * sin(3.14159 * x/nx) + 2 * sin(3.14159 * y/nx)) * Tdiff * (j / N / 3)

                    #d[i, j] = Tmax - (1 - z/nx) * (2 * sin(3.14159 * x/nx) + 2 * sin(3.14159 * y/nx)) * Tdiff * ((N + 1 - j) / N / 3)
                #end
            #end
        end
    end
end



function build_3D_PDE(N, nx, dx, dt, Tmax, Tstart; dense::Bool = true, implicit = false)

    ns = nx^3
    nu = 6

    Q  = 10. * Matrix(LinearAlgebra.I, ns, ns)
    Qf = 10. * Matrix(LinearAlgebra.I, ns, ns)./dt
    R  = 1.0 * Matrix(LinearAlgebra.I, nu, nu)./10

    A, B = build_3D_heating_AB(dx, nx, dt)

    s0 = fill(Tstart, ns)
    sl = fill(200., ns)
    su = fill(550., ns)
    ul = fill(300., nu)
    uu = fill(500., nu)


    #K = .001 * Matrix(I, nu, ns)
    #K = .0001 * ones(nu, ns)
    #K = 1.0 * Matrix(I, nu, ns)
    S = -.001 * Matrix(I, ns, nu)
    Q_scale = 1
    R_scale = 1

    #X, eig, F = ared(A, B, R_scale, Q_scale)
    #K = - F

    if dense
        if implicit
            lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su, ul = ul, uu = uu, S = S, implicit=implicit)
        else
            lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su, ul = ul, uu = uu, S = S)
        end
    else
        lqdm = SparseLQDynamicModel(s0, sparse(A), sparse(B), sparse(Q), sparse(R), N; Qf = sparse(Qf), sl = sl, su = su, ul = ul, uu = uu, S = sparse(S))
    end

    d = zeros(nx^3, N + 1)

    set_d!(d, nx, N, Tmax, Tstart)

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


N = 150
nx = 9
lenx = .02
dt = .5
Tmax = 350.
Tstart = 300.

@time lqdm = build_3D_PDE(N, nx, lenx, dt, Tmax, Tstart; dense = true, implicit = false)


madnlp_options = Dict{Symbol, Any}(
    :kkt_system=>MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    :linear_solver=>LapackCPUSolver,
    :jacobian_constant=>true,
    :hessian_constant=>true,
    :lapack_algorithm=>MadNLP.CHOLESKY,
    #:tol => 1e-6,
    :nlp_scaling=>false,
    :max_iter=>100,
)

ips1 = MadNLP.InteriorPointSolver(lqdm, option_dict=madnlp_options)
sol_ref = MadNLP.optimize!(ips1)
