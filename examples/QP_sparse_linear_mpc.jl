using NLPModels, QuadraticModels, SparseArrays, Random, MadNLP, LinearAlgebra, NLPModelsIpopt

" Build the (sparse) H matrix for quadratic models from Q and R matrices 
 Objective function is 1/2 z^T H z = 1/2 sum(x^T Q x for i in 1:T) + 1/2 sum(u^T R u for i in 1:T)"
function build_H(Q,R, nt; Qf=[])

    if size(Qf)[1] == 0
        Qf = copy(Q)
    end

    ns = size(Q)[1]
    nr = size(R)[1]

    H = sparse([],[],Float64[],(ns*nt + nr*(nt)), (ns*nt + nr*(nt)))

    for i in 1:(nt-1)
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
            row_index = (nt-1)*ns + k
            col_index = (nt-1)*ns + j
            H[row_index, col_index] = Qf[k,j]
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

" Build the (sparse) A matrix for quadratic models from the Ac and B matrices
 where 0 <= Az <= 0 for x_t+1 = Ac* x_t + B* u_t"
function build_A(Ac,B, nt)
    ns = size(Ac)[2]
    nr = size(B)[2]


    A = sparse([],[],Float64[],(ns*(nt-1)), (ns*nt + nr*nt))

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

 
" Get the QuadraticModels.jl QuadraticModel from the Q, R, A, and B matrices
 nt is the number of time steps"
function get_QM(Q, R, A, B, nt; c=zeros(nt*size(Q)[1] + nt*size(R)[1]),
    lvar = fill(-Inf, (nt*size(Q)[1] + nt*size(R)[1])), 
    uvar = fill(Inf, (nt*size(Q)[1] + nt*size(R)[1])),
    x0   = [],
    Qf = [])

    if length(x0) >0 && size(Q)[1] != length(x0)
        error("x0 is not equal to the number of states given in Q")
    end



    H = build_H(Q,R, nt; Qf=Qf)
    A = build_A(A,B, nt)

    len_con = zeros(size(A)[1])

    if length(x0) != 0
        for i in 1:length(x0)
            lvar[i] = x0[i]
            uvar[i] = x0[i]
        end
    end
    
    qp = QuadraticModel(c, H; A = A, lcon = len_con, ucon = len_con, lvar = lvar, uvar = uvar)
    
    return qp

end