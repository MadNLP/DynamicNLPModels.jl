using Revise
using DynamicNLPModels

function build_thinplate(ns, nu, N, dx, dt; d = ones(ns, N + 1).*300, Tbar = 300, dense::Bool = true)
    Q = 1.0Matrix(LinearAlgebra.I, ns, ns)
    Qf= 1.0Matrix(LinearAlgebra.I, ns, ns)/dt
    R = 1.0Matrix(LinearAlgebra.I, nu, nu)/10

    kappa = 400. # thermal conductivity of copper, W/(m-K)
    rho = 8960. # density of copper, kg/m^3
    specificHeat = 386. # specific heat of copper, J/(kg-K)
    thick = .01 # plate thickness in meters
    stefanBoltz = 5.670373e-8 # Stefan-Boltzmann constant, W/(m^2-K^4)
    hCoeff = 1. # Convection coefficient, W/(m^2-K)
    Ta = 300. # The ambient temperature is assumed to be 300 degrees-Kelvin.
    emiss = .5 # emissivity of the plate surface

    conduction_constant =  1 / rho / specificHeat / thick / dx^2

    B = Matrix(LinearAlgebra.I, nu, nu) .* (- dt/(kappa * thick))

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

    s0 = fill(Float64(Tbar), ns)
    sl = fill(Float64(200), ns)
    su = fill(Float64(600), ns)

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
    dQd = zeros(1,1)
    LinearAlgebra.mul!(Qd, block_Q, dvec)
    LinearAlgebra.mul!(dQd, dvec', Qd)


    if dense
        block_A = lqdm.blocks.A
        block_B = lqdm.blocks.B

        As0 = zeros(size(block_A, 1))
        LinearAlgebra.mul!(As0, block_A, s0)
        dQB = zeros(size(block_B, 2))
        LinearAlgebra.mul!(dQB, block_B', Qd)
        dQAs0 = zeros(1,1)
        LinearAlgebra.mul!(dQAs0, Qd', As0)

        lqdm.data.c0 += dQd[1,1] / 2
        lqdm.data.c0 += -dQAs0[1,1]

        lqdm.data.c  += - dQB
    else
        uvec = zeros(nu * N)
        Qdvec = vcat(Qd, uvec)

        lqdm.data.c0 += dQd[1,1] /2
        lqdm.data.c  += - Qdvec
        # Misssing something still; will come back to Monday. 
    end


    return lqdm
end

N = 10
ns = 5
nu = 5
dfunc = (i,k)->100*sin(2*pi*(4*i/N-12*k/ns)) + 400

d = zeros(ns, N+1)

for i in 1:(N+1)
    for j in 1:ns
        d[j,i] = dfunc(i,j)
    end
end

dx = .1
dt = .1
lqdm  = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 300., dense = true)
lqdms = build_thinplate(ns, nu, N, dx, dt; d = d, Tbar = 300., dense = false)

sol_ref = madnlp(lqdm; linear_solver = MadNLPLapackCPU)