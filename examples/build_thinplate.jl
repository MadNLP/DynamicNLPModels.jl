function build_thinplate(ns, nu, N, dx, dt; d = fill(300.0, ns, N+1), Tbar = 300.0, dense::Bool = true, implicit = false,
    sl = -Inf, su = Inf, ul = -Inf, uu = Inf, K = nothing, S = zeros(ns, nu), E = zeros(0, ns), F = zeros(0, nu), gl = zeros(0), gu = zeros(0)
    )
    Q = 1.0 * Matrix(LinearAlgebra.I, ns, ns)
    Qf= 1.0 * Matrix(LinearAlgebra.I, ns, ns)/dt
    R = 1.0 * Matrix(LinearAlgebra.I, nu, nu)/10

    kappa = 400. # thermal conductivity of copper, W/(m-K)
    rho = 8960. # density of copper, kg/m^3
    specificHeat = 386. # specific heat of copper, J/(kg-K)
    thick = .01 # plate thickness in meters
    stefanBoltz = 5.670373e-8 # Stefan-Boltzmann constant, W/(m^2-K^4)
    hCoeff = 1. # Convection coefficient, W/(m^2-K)
    Ta = 300. # The ambient temperature is assumed to be 300 degrees-Kelvin.
    emiss = .5 # emissivity of the plate surface

    conduction_constant =  1 / rho / specificHeat / thick / dx^2

    if ns == nu
        B = Matrix(LinearAlgebra.I, nu, nu) .* (- dt / (kappa * thick))
    elseif nu > ns
        error("number of inputs cannot be greater than number of states")
    else
        # Space out the inputs, u, when nu < ns. These end up mostly evenly spaced
        u_floor   = floor(ns / nu)
        B = zeros(ns, nu)

        index   = 1
        for i in 1:ns
            if index <= nu && i >= index * u_floor
                B[i, index] = -dt / (kappa * thick)
                index += 1
            end
        end
    end


    A = zeros(ns, ns)

    for i in 1:ns
        if i == 1
            A[1, 1] = (-dt * conduction_constant + 1)
            A[1, 2] = (dt * conduction_constant)
        elseif i == ns
            A[ns, ns]     = (-dt * conduction_constant + 1)
            A[ns, ns - 1] = (dt * conduction_constant)
        else
            A[i, i]     = (-2 * dt * conduction_constant + 1)
            A[i, i - 1] = (dt * conduction_constant)
            A[i, i + 1] = (dt * conduction_constant)

        end
    end

    s0 = fill(Tbar, ns)
    sl = fill(sl, ns)
    su = fill(su, ns)
    ul = fill(ul, nu)
    uu = fill(uu, nu)


    if dense
        if implicit
            lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su, ul = ul, uu = uu, E = E, F = F, K = K, S = S, gl = gl, gu = gu, implicit=implicit)
        else
            lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su, ul = ul, uu = uu, E = E, F = F, K = K, S = S, gl = gl, gu = gu)
        end
    else
        lqdm = SparseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su, ul = ul, uu = uu, E = E, F = F, K = K, S = S, gl = gl, gu = gu)
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
