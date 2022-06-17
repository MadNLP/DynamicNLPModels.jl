
"""
    build_QP_JuMP_model(Q,R,A,B,N;...) -> JuMP.Model(...)

Return a `JuMP.jl` Model for the quadratic problem 
    
min 1/2 ( sum_{i=1}^{N-1} s_i^T Q s + sum_{i=1}^{N-1} u^T R u + s_N^T Qf s_n  )
s.t. s_{i+1} = As_i + Bs_i for i = 1,..., N-1

Optional Arguments
- `Qf = []`: matrix multiplied by s_N in objective function (defaults to Q if not given)
- `c = zeros(N*size(Q,1) + N*size(R,1)`:  linear term added to objective funciton, c^T z
- `sl = fill(-Inf, size(Q,1))`: lower bound on state variables
- `su = fill(Inf,  size(Q,1))`: upper bound on state variables
- `ul = fill(-Inf, size(Q,1))`: lower bound on input variables
- `uu = fill(Inf,  size(Q,1))`: upper bound on input variables
- `s0 = []`: initial state of the first state variables

"""
function build_QP_JuMP_model(
        Q,R,A,B, N;
        s0 = [],
        sl = [], 
        su = [],
        ul = [],
        uu = [],
        Qf = [],
        E  = [],
        F  = [],
        gl = [],
        gu = [],
        S  = []
        )

    if size(Qf,1) == 0
        Qf = copy(Q)
    end

    ns = size(Q,1) # Number of states
    nu = size(R,1) # Number of inputs

    NS = 1:ns # set of states
    NN = 1:N # set of times
    NU = 1:nu # set of inputs



    model = Model(MadNLP.Optimizer) # define model


    @variable(model, s[NS, 0:N]) # define states 
    @variable(model, u[NU, 0:(N-1)]) # define inputs



    # Bound states/inputs
    if length(sl) > 0
        for i in NS
            for j in 0:N
                if sl[i] != -Inf
                    @constraint(model, s[i,j] >= sl[i])
                end
            end
        end
    end
    if length(su) > 0
        for i in NS
            for j in 0:N
                if su[i] != Inf
                    @constraint(model, s[i,j] <= su[i])
                end
            end
        end
    end
    if length(ul) > 0
        for i in NU
            for j in 0:(N-1)
                if ul[i] != -Inf
                    @constraint(model,  u[i,j] >= ul[i])
                end
            end
        end
    end
    if length(uu) > 0
        for i in NU
            for j in 0:(N-1)
                if uu[i] != Inf
                    @constraint(model, u[i,j] <= uu[i])
                end
            end
        end
    end

    if length(s0) >0
        for i in NS
            JuMP.fix(s[i,0], s0[i])
        end
    end
    

    # Give constraints from A, B, matrices
    @constraint(model, [t in 0:(N - 1), s1 in NS], s[s1, t + 1] == sum(A[s1, s2] * s[s2, t] for s2 in NS) + sum(B[s1, u1] * u[u1, t] for u1 in NU) )

    # Add E, F constraints
    if length(E) > 0
        for i in 1:size(E,1)
            @constraint(model,[t in 0:(N-1)], gl[i] <= sum(E[i, s1] * s[s1, t] for s1 in NS) + sum(F[i,u1] * u[u1, t] for u1 in NU))
        end
    end

    # Give objective function as xT Q x + uT R u where x is summed over T and u is summed over T-1
    @objective(model,Min,  sum( 1/2 * Q[s1, s2]*s[s1,t]*s[s2,t] for s1 in NS, s2 in NS, t in 0:(N-1)) + 
            sum( 1/2 * R[u1,u2] * u[u1, t] * u[u2,t] for t in 0:(N-1) , u1 in NU, u2 in NU) + 
            sum( 1/2 * Qf[s1,s2] * s[s1,N] * s[s2, N]  for s1 in NS, s2 in NS) +
            sum( S[s1, u1] * s[s1, t] * u[u1, t] for s1 in NS, u1 in NU, t in NN)
            )

    return model
end
