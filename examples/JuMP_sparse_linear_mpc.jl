using Ipopt, JuMP, Random, LinearAlgebra


Random.seed!(10)
Q_org = Random.rand(2,2)
Q = Q_org * transpose(Q_org) + I
R = rand(1) .+ 1

A_org = rand(2,2)
A = A_org * transpose(A_org) + I
B = rand(2,1)


function build_QP_JuMP_model(Q,R,A,B, nt; xlow = [], xhi =[], ulow=[], uhi=[] )

    
    ns = size(Q)[1] # Number of states
    nu = size(R)[1] # Number of inputs

    NS = 1:ns # set of states
    NT = 1:nt # set of times
    NU = 1:nu # set of inputs

    model = Model(Ipopt.Optimizer) # define model

    @variable(model, x[NS, NT]) # define states 
    @variable(model, u[NU, NT]) # define inputs

    # Bound states/inputs
    if length(xlow) > 0
        for i in NS
            for j in NT
                @constraint(model, x[i,j] >= xlow[(j-1)*ns + i])
            end
        end
    end
    
    if length(xhi) > 0
        for i in NS
            for j in NT
                @constraint(model, x[i,j] <= xhi[(j-1)*ns + i])
            end
        end
    end

    if length(ulow) > 0
        for i in NU
            for j in NT
                @constraint(model, u[i,j] >= ulow[(j-1)*nu + i])
            end
        end
    end

    if length(xhi) > 0
        for i in NU
            for j in NT
                @constraint(model, u[i,j] <= xlow[(j-1)*nu + i])
            end
        end
    end

    # Give constraints from A, B, matrices
    @constraint(model, [t in 1:(nt-1), s1 in NS], x[s1, t+1] == sum(A[s1, s2] * x[s2, t] for s2 in NS) + sum(B[s1, u1] * u[u1, t] for u1 in NU) )

    # Give objective function as xT Q x + uT R u where x is summed over T and u is summed over T-1
    @objective(model,Min,  sum( 1/2 * Q[s1, s2]*(x[s1,t])*(x[s2,t]) for s1 in NS, s2 in NS, t in NT) + sum( 1/2 * R[u1,u2] * u[u1, t] * u[u2,t] for t in 1:(nt-1) , u1 in NU, u2 in NU))

    # return model
    return model
end

xlow = fill(1, 6)
xhi  = fill(5, 6)
ulow = fill(0, 3)
uhi  = fill(10,3)

m = build_QP_JuMP_model(Q,R,A,B, 3;xlow = xlow, xhi = xhi, ulow = ulow, uhi = uhi)

optimize!(m)
