using Revise
using LinearAlgebra, Random, SparseArrays, DynamicNLPModels, MadNLP, QuadraticModels

N  = 3 # number of time steps
ns = 2 # number of states
nu = 1 # number of inputs

# generate random Q, R, A, and B matrices
Random.seed!(100)
Q_rand = Random.rand(ns,ns)
Q = Q_rand * transpose(Q_rand) + I
R_rand   = Random.rand(nu,nu)
R    = R_rand * transpose(R_rand) + I

A_rand = rand(ns, ns)
A = A_rand * transpose(A_rand) + I
B = rand(ns, nu)

# generate upper and lower bounds
sl = rand(ns)
ul = rand(nu)
su = sl .+ 10
uu = ul .+ 10
s0 = sl .+ 1


function _build_condensed(
    s0, Q, R, A, B, N;
    Qf = Q)
  
    ns = size(Q,1)
    nu = size(R,1)
  
    # Define block matrices
    block_B = zeros(ns*(N+1), nu*(N))
    block_A = zeros(ns*(N+1), ns)
    block_Q = SparseArrays.sparse([],[], Float64[], ns*(N+1), ns*(N+1))
    block_R = SparseArrays.sparse([],[], Float64[], nu*(N), nu*(N))

    block_A[1:ns, 1:ns] .= Matrix(I, ns, ns)

    # Define matrices for mul!
    A_klast  = copy(A)
    A_k      = similar(A)
    AB_klast = similar(B)
    AB_k     = similar(B)

    # Add diagonal of Bs and fill Q and R block matrices
    for j in 1:(N)
        row_range = (j*ns + 1):((j+1)*ns)
        col_range = ((j-1)*nu+ 1):((j)*nu)
        block_B[row_range, col_range] .= B

        block_Q[((j-1)*ns + 1):(j*ns), ((j-1)*ns + 1):(j*ns)] .= Q
        block_R[((j-1)*nu + 1):(j*nu), ((j-1)*nu + 1):(j*nu)] .= R
    end

    # Fill the A and B matrices
    for i in 1:(N-1)
        if i == 1
            block_A[(ns+1):ns*2, :] .= A
            mul!(AB_k, A, B)
            for k in 1:(N-i)
                row_range = (1 + (k+1) * ns ):((k+2)*ns)
                col_range = (1 + (k-1)*nu):((k)*nu)
                block_B[row_range, col_range] .= AB_k
            end
            AB_klast = copy(AB_k)
        else
            mul!(AB_k, A, AB_klast)
            mul!(A_k, A, A_klast)
            block_A[(ns*i + 1):ns*(i+1),:] .= A_k

            for k in 1:(N-i)
                row_range = (1 + (k+i) * ns):((k+i+1)*ns)
                col_range = (1 + (k-1)*nu):((k)*nu)
                block_B[row_range, col_range] .= AB_k
            end

            AB_klast = copy(AB_k)
            A_klast  = copy(A_k)
        end
    end

    mul!(A_k, A, A_klast)
    block_A[(ns*N +1):ns*(N+1), :] .= A_k

    block_Q[((N)*ns + 1):((N+1)*ns), ((N)*ns + 1):((N+1)*ns)] .= Qf


    QB  = similar(block_B)
    BQB = zeros(nu*N, nu*N)
    mul!(QB, block_Q, block_B)
    mul!(BQB, transpose(block_B), QB)
    axpy!(1, block_R, BQB)

    sTAT = zeros(1, size(block_A,1))
    h    = zeros(1, nu*(N))

    mul!(sTAT, transpose(s0), transpose(block_A))

    mul!(h, sTAT, QB)

    return BQB, h, block_A, block_B, block_Q, block_R
end

H, h, block_A, block_B, block_Q, block_R = _build_condensed(s0, Q, R, A, B, (N))

lqdm = LQDynamicModel(s0, A, B, Q, R, N)


qp = QuadraticModel(vec(h), H)

mlqdm = madnlp(lqdm, max_iter=100)

mqp = madnlp(qp, max_iter=100)

println(mqp.solution)
println(mlqdm.solution[9:11])

println(mqp.objective)
println(mlqdm.objective)