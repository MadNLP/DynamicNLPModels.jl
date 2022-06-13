module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays

export get_QM

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
    _build_H(Q, R, N; Qf = []) -> H

Build the (sparse) `H` matrix from square `Q` and `R` matrices such that 
 z^T H z = sum_{i=1}^{N-1} s_i^T Q s + sum_{i=1}^{N-1} u^T R u + s_N^T Qf s_n . 


# Examples
```julia-repl
julia> Q = [1 2; 2 1]; R = ones(1,1); _build_H(Q, R, 2)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
 1.0  2.0   ⋅    ⋅    ⋅    ⋅ 
 2.0  1.0   ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅   1.0  2.0   ⋅    ⋅
  ⋅    ⋅   2.0  1.0   ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅     ⋅
```

If `Qf` is not given, then `Qf` defaults to `Q`
"""
function _build_H(
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
    _build_J(A, B, N) -> J

Build the (sparse) `J` matrix or a linear model from `A` and `B` matrices such that
0 <= Jz <= 0 is equivalent to s_{i+1} = As_i + Bs_i for i = 1,..., N-1

# Examples
```julia-repl
julia> A = [1 2 ; 3 4]; B = [5 6; 7 8]; _build_J(A,B,3)
4×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 20 stored entries:
 1.0  2.0  -1.0    ⋅     ⋅     ⋅   5.0  6.0   ⋅    ⋅    ⋅    ⋅
 3.0  4.0    ⋅   -1.0    ⋅     ⋅   7.0  8.0   ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    1.0   2.0  -1.0    ⋅    ⋅    ⋅   5.0  6.0   ⋅    ⋅
  ⋅    ⋅    3.0   4.0    ⋅   -1.0   ⋅    ⋅   7.0  8.0   ⋅    ⋅
```
"""
function _build_J(A,B, N)
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
    get_QM(Q,R,A,B,N; ...) -> QuadraticModels.QuadraticModel(...)

Returns a `QuadraticModel` from the matrices `Q`, `R`, `A`, and `B` with linear constraints over `N` time steps. 

QuadraticModel has the form of  

min 1/2 z^T H z 
s.t. 0 <= Jx <= 0

which is equivalent to the problem of 

min 1/2 ( sum_{i=1}^{N-1} s_i^T Q s + sum_{i=1}^{N-1} u^T R u + s_N^T Qf s_n  )
s.t. s_{i+1} = As_i + Bs_i for i = 1,..., N-1

# Optional Arguments
- `Qf = []`: matrix multiplied by s_N in objective function (defaults to Q if not given)
- `c = zeros(N*size(Q,1) + N*size(R,1)`:  linear term added to objective funciton, c^T z
- `sl = fill(-Inf, size(Q,1))`: lower bound on state variables
- `su = fill(Inf,  size(Q,1))`: upper bound on state variables
- `ul = fill(-Inf, size(Q,1))`: lower bound on input variables
- `uu = fill(Inf,  size(Q,1))`: upper bound on input variables
- `s0 = []`: initial state of the first state variables

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



    H = _build_H(Q,R, N; Qf=Qf)
    J = _build_J(A,B, N)

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
