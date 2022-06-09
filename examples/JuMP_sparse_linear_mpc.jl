using Ipopt, JuMP, Random, LinearAlgebra



"Build a JuMP model to minimize 1/2 sum(x^T Q x for i in 1:nt) + 1/2 sum(u^T R u for i in 1:nt-1)
subject to x(t+1) = Ax(t) + Bu(t) and lvar <= [x(1), ..., x(t), u(1), ..., u(t)] <= uvar"
function build_QP_JuMP_model(Q,R,A,B, nt;
        lvar = [], 
        uvar = [],
        Qf  = [])

    if size(Qf)[1] ==0
        Qf = copy(Q)
    end

    ns = size(Q)[1] # Number of states
    nu = size(R)[1] # Number of inputs

    NS = 1:ns # set of states
    NT = 1:nt # set of times
    NU = 1:nu # set of inputs



    model = Model(Ipopt.Optimizer) # define model

    @variable(model, x[NS, NT]) # define states 
    @variable(model, u[NU, NT]) # define inputs

    # Bound states/inputs
    if length(lvar) > 0
        for j in NT
            for i in NS
                @constraint(model, x[i,j] >= lvar[(j-1)*ns + i])
            end

            for i in NU
                @constraint(model, u[i,j] >= lvar[ns*nt + (j-1)*nu + i])
            end

        end
    end


    if length(uvar) > 0
        for j in NT
            for i in NS
                @constraint(model, x[i,j] <= uvar[(j-1)*ns + i])
            end

            for i in NU
                @constraint(model, u[i,j] <= uvar[ns*nt + (j-1)*nu + i])
            end

        end
    end

    

    # Give constraints from A, B, matrices
    @constraint(model, [t in 1:(nt-1), s1 in NS], x[s1, t+1] == sum(A[s1, s2] * x[s2, t] for s2 in NS) + sum(B[s1, u1] * u[u1, t] for u1 in NU) )

    # Give objective function as xT Q x + uT R u where x is summed over T and u is summed over T-1
    @objective(model,Min,  sum( 1/2 * Q[s1, s2]*(x[s1,t])*(x[s2,t]) for s1 in NS, s2 in NS, t in 1:(nt-1)) + 
            sum( 1/2 * R[u1,u2] * u[u1, t] * u[u2,t] for t in 1:(nt-1) , u1 in NU, u2 in NU) + 
            sum(1/2 * Qf[s1,s2] * x[s1,nt] * x[s2, nt]  for s1 in NS, s2 in NS))

    # return model
    return model
end