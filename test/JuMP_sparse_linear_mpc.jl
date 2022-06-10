using Ipopt, JuMP, Random, LinearAlgebra



"""
Build a JuMP model to minimize 1/2 sum(x^T Q x for i in 1:N) + 1/2 sum(u^T R u for i in 1:(N-1))
subject to x(t+1) = Ax(t) + Bu(t) and lvar <= [x(1), ..., x(t), u(1), ..., u(t)] <= uvar
"""
function build_QP_JuMP_model(
        Q,R,A,B, N;
        s0 = [],
        sl = [], 
        su = [],
        ul = [],
        uu = [],
        Qf = []
        )

    if size(Qf,1) == 0
        Qf = copy(Q)
    end

    ns = size(Q,1) # Number of states
    nu = size(R,1) # Number of inputs

    NS = 1:ns # set of states
    NN = 1:N # set of times
    NU = 1:nu # set of inputs



    model = Model(Ipopt.Optimizer) # define model


        @variable(model, s[NS, NN]) # define states 
        @variable(model, u[NU, NN]) # define inputs



    # Bound states/inputs
    if length(sl) > 0
        for i in NS
            for j in NN
                @constraint(model, s[i,j] >= sl[i])
            end
        end
    end
    if length(su) > 0
        for i in NS
            for j in NN
                @constraint(model, s[i,j] <= su[i])
            end
        end
    end
    if length(ul) > 0
        for i in NU
            for j in NN
                @constraint(model,  u[i,j] >= ul[i])
            end
        end
    end
    if length(uu) > 0
        for i in NU
            for j in NN
                @constraint(model, u[i,j] <= uu[i])
            end
        end
    end

    if length(s0) >0
        for i in NS
            JuMP.fix(s[i,1], s0[i])
        end
    end
    

    # Give constraints from A, B, matrices
    @constraint(model, [t in 1:(N-1), s1 in NS], s[s1, t+1] == sum(A[s1, s2] * s[s2, t] for s2 in NS) + sum(B[s1, u1] * u[u1, t] for u1 in NU) )

    # Give objective function as xT Q x + uT R u where x is summed over T and u is summed over T-1
    @objective(model,Min,  sum( 1/2 * Q[s1, s2]*(s[s1,t])*(s[s2,t]) for s1 in NS, s2 in NS, t in 1:(N-1)) + 
            sum( 1/2 * R[u1,u2] * u[u1, t] * u[u2,t] for t in 1:(N-1) , u1 in NU, u2 in NU) + 
            sum( 1/2 * Qf[s1,s2] * s[s1,N] * s[s2, N]  for s1 in NS, s2 in NS))

    # return model
    return model
end
#=
function build_H(
    Q, R, N;
    Qf = [])
    if size(Qf,1) == 0
        Qf = copy(Q)
    end

    ns = size(Q,1)
    nr = size(R,1)

    H = SparseArrays.sparse([],[],Float64[],(ns*N + nr*(N)), (ns*N + nr*(N)))

    for i in 1:(N-1)
        for j in 1:ns
            for k in 1:ns
                row_index = (i-1)*ns + k
                col_index = (i-1)*ns + j
                H[row_index, col_index] = Q[k,j]

            end
        end
    end

    for j in 1:ns
        for k in 1:ns
            row_index = (N-1)*ns + k
            col_index = (N-1)*ns + j
            H[row_index, col_index] = Qf[k,j]
        end
    end


    for i in 1:(N-1)
        for j in 1:nr
            for k in 1:nr
                row_index = ns*N + (i-1) * nr + k
                col_index = ns*N + (i-1) * nr + j
                H[row_index, col_index] = R[k,j]
            end
        end
    end

    return H
end


"""
Build the (sparse) A matrix for quadratic models from the Ac and B matrices
where 0 <= Jz <= 0 for x_t+1 = Ac* x_t + B* u_t
"""

function build_J(A,B, N)
    ns = size(A,2)
    nr = size(B,2)


    J = SparseArrays.sparse([],[],Float64[],(ns*(N-1)), (ns*N + nr*N))    

    for i in 1:(N-1)
        for j in 1:ns
            row_index = (i-1)*ns + j
            J[row_index, (i*ns + j)] = -1
            for k in 1:ns
                col_index = (i-1)*ns + k
                J[row_index, col_index] = A[j,k]
            end

            for k in 1:nr
                col_index = (N*ns) + (i-1)*nr + k
                J[row_index, col_index] = B[j,k]    
            end
        end
    end

    return J
end


 
"""
Get the QuadraticModels.jl QuadraticModel from the Q, R, A, and B matrices
nt is the number of time steps
"""
function get_QM(
    Q, R, A, B, N;
    c    = zeros(N*size(Q,1) + N*size(R,1)),
    sl   = fill(-Inf, N*size(Q,1)),
    su   = fill(Inf, N*size(Q,1)),
    ul   = fill(-Inf, N*size(R,1)),
    uu   = fill(Inf, N*size(R,1)),
    s0   = [],
    Qf   = [])

    if length(s0) >0 && size(Q,1) != length(s0)
        error("s0 is not equal to the number of states given in Q")
    end



    H = build_H(Q,R, N; Qf=Qf)
    J = build_J(A,B, N)

    con = zeros(size(A,1))


    if length(s0) != 0
        lvar = copy(s0)
        uvar = copy(s0)
    else
        lvar = copy(su)
        uvar = copy(sl)
    end

    for i in 1:(N-1)
        lvar = vcat(lvar, sl)
        uvar = vcat(uvar, su)
    end

    for i in 1:(N)
        lvar = vcat(lvar, ul)
        uvar = vcat(uvar, su)
    end
        
    qp = QuadraticModels.QuadraticModel(c, H; A = A, lcon = con, ucon = con, lvar = lvar, uvar = uvar)
    
    return qp

end
=#