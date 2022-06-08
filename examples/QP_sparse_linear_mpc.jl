using NLPModels, QuadraticModels, SparseArrays, Random, MadNLP, LinearAlgebra, NLPModelsIpopt

# Build the (sparse) H matrix for quadratic models from Q and R matrices 
# Objective function is z^T H z = sum(x^T Q x for i in 1:T) + sum(u^T R u for i in 1:T)
function build_H(Q,R, nt)
    ns = size(Q)[1]
    nr = size(R)[1]

    H = sparse([],[],Float64[],(ns*nt + nr*(nt)), (ns*nt + nr*(nt)))

    for i in 1:nt
        for j in 1:ns
            for k in 1:ns
                row_index = (i-1)*ns + k
                col_index = (i-1)*ns + j
                H[row_index, col_index] = Q[k,j]

            end
        end
    end

    for i in 1:(nt-1)
        for j in 1:nr
            for k in 1:nr
                row_index = ns*nt + (i-1) * nr + k
                col_index = ns*nt + (i-1) * nr + j
                H[row_index, col_index] = R[k,j]
            end
        end
    end

    return H
end

# Build the (sparse) A matrix for quadratic models from the Ac and B matrices
# where 0 <= Az <= 0 for x_t+1 = Ac* x_t + B* u_t
function build_A(Ac,B, nt)
    ns = size(Ac)[2]
    nr = size(B)[2]


    A = sparse([],[],Float64[],(ns*nt), (ns*nt + nr*nt))

    for i in 1:(nt-1)
        for j in 1:ns
            row_index = (i-1)*ns + j
            A[row_index, (i*ns + j)] = -1
            for k in 1:ns
                col_index = (i-1)*ns + k
                A[row_index, col_index] = Ac[j,k]
            end

            for k in 1:nr
                col_index = (nt*ns) + (i-1)*nr + k
                A[row_index, col_index] = B[j,k]    
            end
        end
    end

    return A
end

 
# Get the QuadraticModels.jl QuadraticModel from the Q, R, A, and B matrices
# nt is the number of time steps
function get_QM(Q, R, A, B, nt;
    lvar = fill(-Inf, (nt*size(Q)[1] + nt*size(R)[1])), 
    uvar = fill(Inf, (nt*size(Q)[1] + nt*size(R)[1]))   )

    ns = size(Q)[1]
    nu = size(R)[1]

    H = build_H(Q,R, nt)
    A = build_A(A,B, nt)


    c       = zeros(ns*nt + nu*nt)
    len_con = zeros(size(A)[1])



    qp = QuadraticModel(c, H; A = A, lcon = len_con, ucon = len_con, lvar = lvar, uvar = uvar)
    
    return qp

end


# Build Q, R, A, and B matrices for 2 states and 1 input
Random.seed!(10)
Q_org = Random.rand(2,2)
Q = Q_org * transpose(Q_org) + I
R = rand(1) .+ 1

A_org = rand(2,2)
A = A_org * transpose(A_org) + I
B = rand(2,1)


# Set the lower value of state variables to be 1, upper value to be 5
# Set the lower value of input variables to be 0, upper value to be 10
lvar = fill(-Inf, 9)
uvar = fill(Inf, 9)
for i in 1:6
    lvar[i] = 1
    uvar[i] = 5
end
for i in 7:9
    lvar[i] = 0
    uvar[i] = 10
end


qp = get_QM(Q, R, A, B, 3; lvar=lvar, uvar = uvar)

madnlp(qp, max_iter = 100)