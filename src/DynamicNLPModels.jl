module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays

export build_H, build_J, get_QM

mutable struct LQDynData{VT,MT}
    N::Int
    nx::Int
    nu::Int
    
    A::MT
    B::MT
    Q::MT
    R::MT
    Qf::MT
    
    x0::VT
    xl::VT
    xu::VT
    ul::VT
    uu::VT
end

function LQDynData(
    N, x0, A, B, Q, R;
    Qf = Q,
    xl = (similar(x0) .= -Inf),
    xu = (similar(x0) .=  Inf),
    ul = (similar(x0,nu) .= -Inf),
    uu = (similar(x0,nu) .=  Inf)
    )

end

function QPData(dnlp::LQDynData{VT,MT}) where {VT,MT}
    
end

""" 
Build the (sparse) H matrix for quadratic models from Q and R matrices 
Objective function is 1/2 z^T H z = 1/2 sum(x^T Q x for i in 1:T) + 1/2 sum(u^T R u for i in 1:(T-1))
"""
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
    sl   = fill(-Inf, size(Q,1)),
    su   = fill(Inf,  size(Q,1)),
    ul   = fill(-Inf, size(R,1)),
    uu   = fill(Inf,  size(R,1)),
    s0   = [],
    Qf   = [])

    if length(s0) >0 && size(Q,1) != length(s0)
        error("s0 is not equal to the number of states given in Q")
    end



    H = build_H(Q,R, N; Qf=Qf)
    J = build_J(A,B, N)

    con = zeros(size(J,1))


    if length(s0) != 0
        lvar = copy(s0)
        uvar = copy(s0)
    else
        lvar = copy(sl)
        uvar = copy(su)
    end

    for i in 1:(N-1)
        lvar = vcat(lvar, sl)
        uvar = vcat(uvar, su)
    end

    for i in 1:(N)
        lvar = vcat(lvar, ul)
        uvar = vcat(uvar, uu)
    end

    qp = QuadraticModels.QuadraticModel(c, H; A = J, lcon = con, ucon = con, lvar = lvar, uvar = uvar)
    
    return qp

end

end # module
