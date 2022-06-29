function build_thinplate(ns, nu, N, dx, dt; d = fill(300.0, ns, N+1), Tbar = 300.0, dense::Bool = true, sl = -Inf, su = Inf)
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


    if dense
        lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su)
    else
        lqdm = SparseLQDynamicModel(s0, A, B, Q, R, N; Qf = Qf, sl = sl, su = su)
    end

    if dense
        block_Q = lqdm.blocks.Q
    else
        block_Q = SparseArrays.sparse([],[],eltype(Q)[], ns * (N + 1), ns * (N + 1))
        for i in 1:N
            block_Q[(1 + (i - 1) * ns):(ns * i), (1 + (i - 1) * ns):(ns * i)] = Q
        end
        block_Q[(1 + ns * N):end, (1 + ns * N):end] = Qf
    end
    
    
    dvec = vec(d)
    Qd  = similar(dvec)
    dQd = zeros(1)
    LinearAlgebra.mul!(Qd, block_Q, dvec)
    LinearAlgebra.mul!(dQd, dvec', Qd)

    # Add c and c0 that result from (x-d)^T Q (x-d) in the objective function
    if dense
        block_A = lqdm.blocks.A
        block_B = lqdm.blocks.B

        As0 = zeros(size(block_A, 1))
        LinearAlgebra.mul!(As0, block_A, s0)
        dQB = zeros(size(block_B, 2))
        LinearAlgebra.mul!(dQB, block_B', Qd)
        dQAs0 = zeros(1)
        LinearAlgebra.mul!(dQAs0, Qd', As0)

        lqdm.data.c0 += dQd[1,1] / 2
        lqdm.data.c0 += -dQAs0[1,1]

        lqdm.data.c  += - dQB
    else
        uvec = zeros(nu * N)
        Qdvec = vcat(Qd, uvec)

        lqdm.data.c0 += dQd[1,1] /2
        lqdm.data.c  += - Qdvec
    end

    return lqdm
end