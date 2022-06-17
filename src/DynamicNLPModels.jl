module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC

export get_QM, LQDynamicData, LQDynamicModel

abstract type AbstractLQDynData{T,V} end
"""
    LQDynamicData{T,V,M} <: AbstractLQDynData{T,V}

A struct to represent the features of the optimization problem 

```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                gl \\le E s_i + F u_i \\le gu for i = 0, 1, ..., N-1
                sl \\le s \\le su
                ul \\le u \\le uu
                s_0 = s0
```
--- 
Attributes include:
- `s0`: initial state of system
- `A` : constraint matrix for system states
- `B` : constraint matrix for system inputs
- `Q` : objective function matrix for system states from 1:(N-1)
- `R` : objective function matrix for system inputs from 1:(N-1)
- `N` : number of time steps
- `Qf`: objective function matrix for system state at time N; defaults to Q
- `ns`: number of state variables
- `nu`: number of input varaibles
- `E` : constraint matrix for state variables
- `F` : constraint matrix for input variables
- `sl`: vector of lower bounds on state variables
- `su`: vector of upper bounds on state variables
- `ul`: vector of lower bounds on input variables
- `uu`: vector of upper bounds on input variables
- `gl`: vector of lower bounds on constraints
- `gu`: vector of upper bounds on constraints

see also `LQDynamicData(s0, A, B, Q, R, N; ...)`
"""
struct LQDynamicData{T,V,M} <: AbstractLQDynData{T,V}
    s0::V
    A::M
    B::M
    Q::M
    R::M
    N

    Qf::M
    S::M
    ns::Int
    nu::Int
    E::M
    F::M

    sl::V
    su::V
    ul::V
    uu::V
    gl::V
    gu::V
end

"""
    LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, V, M}
A constructor for building an object of type `LQDynamicData` for the optimization problem 
```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                gl \\le E s_i + F u_i \\le gu for i = 0, 1, ..., N-1
                sl \\le s \\le su
                ul \\le u \\le uu
                s_0 = s0
```
---
- `s0`: initial state of system
- `A` : constraint matrix for system states
- `B` : constraint matrix for system inputs
- `Q` : objective function matrix for system states from 1:(N-1)
- `R` : objective function matrix for system inputs from 1:(N-1)
- `N` : number of time steps
The following attributes of the `LQDynamicData` type are detected automatically from the length of s0 and size of R
- `ns`: number of state variables
- `nu`: number of input varaibles
The following keyward arguments are also accepted
- `Qf = Q`: objective function matrix for system state at time N; dimensions must be ns x ns
- `E  = zeros(0, ns)` : constraint matrix for state variables
- `F  = zeros(0, nu)` : constraint matrix for input variables
- `sl = fill(-Inf, ns)`: vector of lower bounds on state variables
- `su = fill(Inf, ns)` : vector of upper bounds on state variables
- `ul = fill(-Inf, nu)`: vector of lower bounds on input variables
- `uu = fill(Inf, nu)` : vector of upper bounds on input variables
- `gl = fill(-Inf, size(E, 1))` : vector of lower bounds on constraints
- `gu = fill(Inf, size(E, 1))`  : vector of upper bounds on constraints
"""
function LQDynamicData(
    s0::V,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;

    Qf::M = Q, 
    S::M  = zeros(size(Q, 1), size(R, 1)),
    E::M  = zeros(0, length(s0)),
    F::M  = zeros(0, size(R, 1)),

    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R,1)) .= -Inf),
    uu::V = (similar(s0, size(R,1)) .=  Inf),
    gl::V = fill(-Inf, size(E, 1)),
    gu::V = fill(Inf, size(F, 1))
    ) where {T,V <: AbstractVector{T}, M <: AbstractMatrix{T}}

    if size(Q, 1) != size(Q, 2) 
        error("Q matrix is not square")
    end
    if size(R, 1) != size(R, 1)
        error("R matrix is not square")
    end
    if size(A, 2) != length(s0)
        error("Number of columns of A are not equal to the number of states")
    end
    if size(B, 2) != size(R, 1)
        error("Number of columns of B are not equal to the number of inputs")
    end
    if length(s0) != size(Q, 1)
        error("size of Q is not consistent with length of x0")
    end

    if !(sl  <= su)
        error("lower bound(s) on x is > upper bound(s)")
    end
    if !(ul <= uu)
        error("lower bound(s) on u is > upper bound(s)")
    end
    if !(s0 >= sl) || !(s0 <= su)
        error("x0 is not within the given upper and lower bounds")
    end

    if size(E, 1) != size(F, 1)
        error("E and F have different numbers of rows")
    end
    if !(gl <= gu)
        error("lower bound(s) on Es + Fu is > upper bound(s)")
    end
    if size(E, 2) != size(Q, 1)
        error("Dimensions of E are not the same as number of states")
    end
    if size(F, 2) != size(R, 1) 
        error("Dimensions of F are not the same as the number of inputs")
    end
    if length(gl) != size(E, 1)
        error("Dimensions of gl do not match E and F")
    end
    if length(gu) != size(E, 1)
        error("Dimensions of gu do not match E and F")
    end

    if size(S, 1) != size(Q, 1) || size(S, 2) != size(R, 1)
        error("Dimensions of S do not match dimensions of Q and R")
    end



    ns= size(Q,1)
    nu= size(R,1)

    LQDynamicData{T,V,M}(
        s0, A, B, Q, R, N,
        Qf, S, ns, nu, E, F,
        sl, su, ul, uu, gl, gu
    )
end



abstract type AbstractDynamicModel{T,V} <: QuadraticModels.AbstractQuadraticModel{T, V} end

mutable struct LQDynamicModel{T, V, M1, M2, M3} <:  AbstractDynamicModel{T,V} 
  meta::NLPModels.NLPModelMeta{T, V}
  counters::NLPModels.Counters
  data::QuadraticModels.QPData{T, V, M1, M2}
  dynamic_data::LQDynamicData{T, V, M3}
  condense::Bool
end

"""
    LQDynamicModel(dnlp::LQDynamicData; condense=false)      -> LQdynamicModel
    LQDynamicModel(s0, A, B, Q, R, N; condense = false, ...) -> LQDynamicModel
A constructor for building a `LQDynamicModel <: QuadraticModels.AbstractQuadraticModel` from `LQDynamicData`
Input data is for the problem of the form 
```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                gl \\le E s_i + F u_i \\le gu for i = 0, 1, ..., N-1            
                sl \\le s \\le su
                ul \\le u \\le uu
                s_0 = s0
```
---

If `condense=false`, data is converted to the form 

```math
    minimize    \\frac{1}{2} z^T H z 
    subject to  lcon \\le Jz \\le ucon
                lvar \\le z \\le uvar
```
Resulting `H` and `J` matrices are stored as `QuadraticModels.QPData` within the `LQDynamicModel` struct and 
variable and constraint limits are stored within `NLPModels.NLPModelMeta`

---

If `condense=true`, data is converted to the form 

```math
    minimize    \\frac{1}{2} u^T H u + h^T u + h0 
    subject to  Jz \\le g
                ul \\le u \\le uu
```

Resulting `H`, `J`, `h`, and `h0` matrices are stored within `QuadraticModels.QPData` as `H`, `A`, `c`, and `c0` attributes respectively
"""
function LQDynamicModel(dnlp::LQDynamicData{T,V,M}; condense = false) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}}

    if condense==false
        _build_sparse_lq_dynamic_model(dnlp)
    else
        _build_condensed_lq_dynamic_model(dnlp)
    end


end


function LQDynamicModel(
    s0::V,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;
    Qf::M = Q, 
    S::M  = zeros(size(Q, 1), size(R, 1)),
    E::M  = zeros(0, length(s0)),
    F::M  = zeros(0, size(R, 1)),
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0,size(R, 1)) .= -Inf),
    uu::V = (similar(s0,size(R, 1)) .=  Inf),
    gl::V = fill(-Inf, size(E, 1)),
    gu::V = fill(Inf, size(F, 1)),
    condense=false
    ) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}

    dnlp = LQDynamicData(s0, A, B, Q, R, N; Qf = Qf, S = S, E = E, F = F, sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    
    LQDynamicModel(dnlp; condense=condense)

end

function _build_sparse_lq_dynamic_model(dnlp::LQDynamicData{T,V,M}) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}}
    s0 = dnlp.s0
    A  = dnlp.A
    B  = dnlp.B
    Q  = dnlp.Q
    R  = dnlp.R
    N  = dnlp.N

    Qf = dnlp.Qf
    S  = dnlp.S
    ns = dnlp.ns
    nu = dnlp.nu
    E  = dnlp.E
    F  = dnlp.F

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu



    H   = _build_H(Q, R, N; Qf = Qf, S = S)
    J1  = _build_sparse_J1(A, B, N)
    J2  = _build_sparse_J2(E, F, N)
    J   = vcat(J1, J2)
    c0 = 0.0

    
    nvar = ns * (N + 1) + nu * N
    c  = zeros(nvar)
    
    lvar  = zeros(nvar)
    uvar  = zeros(nvar)

    lvar[1:ns] .= s0
    uvar[1:ns] .= s0


    ucon  = zeros(ns * N + N * length(gl))
    lcon  = zeros(ns * N + N * length(gl))

    ncon  = size(J, 1)
    nnzj = length(J.rowval)
    nnzh = length(H.rowval)

    for i in 1:N
        lvar[(i * ns + 1):((i + 1) * ns)] .= sl
        uvar[(i * ns + 1):((i + 1) * ns)] .= su

        lcon[(ns * N + 1 + (i -1) * length(gl)):(ns * N + i * length(gl))] .= gl
        ucon[(ns * N + 1 + (i -1) * length(gl)):(ns * N + i * length(gl))] .= gu
    end

    for j in 1:N
        lvar[((N + 1) * ns + (j - 1) * nu + 1):((N + 1) * ns + j * nu)] .= ul
        uvar[((N + 1) * ns + (j - 1) * nu + 1):((N + 1) * ns + j * nu)] .= uu
    end
    


    LQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        lvar = lvar,
        uvar = uvar, 
        ncon = ncon,
        lcon = lcon,
        ucon = ucon,
        nnzj = nnzj,
        nnzh = nnzh,
        lin = 1:ncon,
        islp = (ncon == 0);
        ),
        NLPModels.Counters(),
        QuadraticModels.QPData(
        c0, 
        c,
        H,
        J
        ),
        dnlp,
        false
    )

end

function _build_condensed_lq_dynamic_model(dnlp::LQDynamicData{T,V,M}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    s0 = dnlp.s0
    A  = dnlp.A
    B  = dnlp.B
    Q  = dnlp.Q
    R  = dnlp.R
    N  = dnlp.N

    Qf = dnlp.Qf
    S  = dnlp.S
    ns = dnlp.ns
    nu = dnlp.nu
    E  = dnlp.E
    F  = dnlp.F

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu

    
    condensed_blocks = _build_condensed_blocks(s0, Q, R, A, B, N; Qf = Qf, S = S)

    block_A = condensed_blocks.block_A
    block_B = condensed_blocks.block_B
    H       = condensed_blocks.H
    c       = condensed_blocks.c
    c0      = condensed_blocks.c0

    lvar = zeros(nu * N)
    uvar = zeros(nu * N)

    for i in 1:(N)
        lvar[((i - 1) * nu + 1):(i * nu)] = ul
        uvar[((i - 1) * nu + 1):(i * nu)] = uu
    end


    nE1 = size(E, 1)
    nE2 = size(E, 2)
    nF1 = size(F, 1)
    nF2 = size(F, 2)
    ng  = length(gl)

    
    block_E  = zeros(nE1 * N, nE2 * (N + 1))
    block_F  = zeros(nF1 * N, nF2 * N)
    block_gl = zeros(ng * N, 1)
    block_gu = zeros(ng * N, 1)
  
    for i in 1:N
        block_E[((i - 1) * nE1 + 1):(i * nE1), ((i - 1) * nE2 + 1):(i * nE2)] .= E
        block_F[((i - 1) * nF1 + 1):(i * nF1), ((i - 1) * nF2 + 1):(i * nF2)] .= F
        block_gl[((i - 1) * ng  + 1):(i * ng)]  .= gl
        block_gu[((i - 1) * ng  + 1):(i * ng)]  .= gu
    end


    condensed_blocks_G = _build_condensed_G(block_A, block_B, block_E, block_F, block_gl, block_gu, s0, N)

    J1    = condensed_blocks_G.J
    lcon1 = condensed_blocks_G.lcon
    ucon1 = condensed_blocks_G.ucon
    As0   = condensed_blocks_G.As0

    bool_vec        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds = sum(bool_vec)

    if num_real_bounds == length(sl)
        J2         = block_B[(1 + ns):end,:]
        As0_bounds = As0[(1 + ns):end,1]
    else        
        J2         = zeros(num_real_bounds * N, nu * N)
        As0_bounds = zeros(num_real_bounds * N, 1)
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds):(i * num_real_bounds)
            J2[row_range, :] = block_B[(1 + ns * i): (ns * (i + 1)), :][bool_vec, :]
            As0_bounds[row_range, :] = As0[(1 + ns * i):(ns * (i + 1)), :][bool_vec, :]
        end


        sl = sl[bool_vec]
        su = su[bool_vec]
    end


    lcon  = zeros(length(lcon1) + size(J2,1))
    ucon  = zeros(length(ucon1) + size(J2,1))

    lcon2 = zeros(size(J2, 1),1)
    ucon2 = zeros(size(J2, 1),1)

    lcon[1:length(lcon1)] .= lcon1
    ucon[1:length(ucon1)] .= ucon1


    for i in 1:N
        lcon2[((i - 1) * length(su) + 1):(i * length(su)),1] = sl
        ucon2[((i - 1) * length(su) + 1):(i * length(su)),1] = su
    end

    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)
    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)

    lcon[(length(lcon1) + 1):end] .= lcon2
    ucon[(length(ucon1) + 1):end] .= ucon2

    J = vcat(J1, J2)

    nvar = nu * N
    nnzj = size(J, 1) * size(J, 2)
    nnzh = sum(LinearAlgebra.LowerTriangular(H) .!= 0)
    ncon = size(J, 1)


    LQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        lvar = lvar,
        uvar = uvar, 
        ncon = ncon,
        lcon = lcon,
        ucon = ucon,
        nnzj = nnzj,
        nnzh = nnzh,
        lin = 1:ncon,
        islp = (ncon == 0);
        ),
        NLPModels.Counters(),
        QuadraticModels.QPData(
        c0, 
        c,
        H,
        J
        ),
        dnlp,
        true
    )
end


function _build_condensed_blocks(
    s0, Q, R, A, B, N;
    Qf = Q, 
    S = zeros(size(Q, 1), size(R, 1)))
  
    ns = size(Q, 1)
    nu = size(R, 1)
  
    # Define block matrices
    block_B = zeros(ns * (N + 1), nu * N)
    block_A = zeros(ns * (N + 1), ns)
    block_Q = SparseArrays.sparse([],[], eltype(Q)[], ns * (N + 1), ns * (N + 1))
    block_R = SparseArrays.sparse([],[], eltype(R)[], nu * N, nu * N)
  
    block_A[1:ns, 1:ns] .= Matrix(LinearAlgebra.I, ns, ns)
  
    # Define matrices for mul!
    A_klast  = copy(A)
    A_k      = zeros(size(A))
    AB_klast = zeros(size(B))
    AB_k     = zeros(size(B))
  
    # Add diagonal of Bs and fill Q and R block matrices
    for j in 1:N
        row_range = (j * ns + 1):((j + 1) * ns)
        col_range = ((j - 1) * nu+ 1):(j * nu)
        block_B[row_range, col_range] .= B
  
        block_Q[((j - 1) * ns + 1):(j * ns), ((j - 1) * ns + 1):(j * ns)] .= Q
        block_R[((j - 1) * nu + 1):(j * nu), ((j - 1) * nu + 1):(j * nu)] .= R
    end
  
    # Fill the A and B matrices
    for i in 1:(N - 1)
        if i == 1
            block_A[(ns + 1):ns*2, :] .= A
            LinearAlgebra.mul!(AB_k, A, B)
            for k in 1:(N-i)
                row_range = (1 + (k + 1) * ns):((k + 2) * ns)
                col_range = (1 + (k - 1) * nu):(k * nu)
                block_B[row_range, col_range] .= AB_k
            end
            AB_klast = copy(AB_k)
        else
            LinearAlgebra.mul!(AB_k, A, AB_klast)
            LinearAlgebra.mul!(A_k, A, A_klast)
            block_A[(ns * i + 1):ns * (i + 1),:] .= A_k
  
            for k in 1:(N-i)
                row_range = (1 + (k + i) * ns):((k + i + 1) * ns)
                col_range = (1 + (k - 1) * nu):(k * nu)
                block_B[row_range, col_range] .= AB_k
            end
  
            AB_klast = copy(AB_k)
            A_klast  = copy(A_k)
        end
    end
  

    LinearAlgebra.mul!(A_k, A, A_klast)

    block_A[(ns * N + 1):ns * (N + 1), :] .= A_k
    block_Q[(ns * N + 1):((N + 1) * ns), (N * ns + 1):((N + 1) * ns)] .= Qf
  
    # build quadratic term
    QB  = similar(block_B)
    BQB = zeros(nu * N, nu * N)

    LinearAlgebra.mul!(QB, block_Q, block_B)
    LinearAlgebra.mul!(BQB, transpose(block_B), QB)
    LinearAlgebra.axpy!(1, block_R, BQB)
    
    # build linear term 
    h    = zeros(1, nu * N)
    s0TAT = zeros(1, size(block_A,1))
    LinearAlgebra.mul!(s0TAT, transpose(s0), transpose(block_A))
    LinearAlgebra.mul!(h, s0TAT, QB)
  
    # build constant term 
    QAs = zeros(size(s0TAT,2), size(s0TAT,1))
    h0  = zeros(1,1)
    LinearAlgebra.mul!(QAs, block_Q, transpose(s0TAT))
    LinearAlgebra.mul!(h0, s0TAT, QAs)
  
    c0 = h0[1,1] / 2
  
    if S != zeros(size(Q, 1), size(R, 1))
        block_S = zeros(ns * (N + 1), nu * N)

        for i in 1:N
            row_range = (1 + ns * (i - 1)):(ns * i)
            col_range = (1 + nu * (i - 1)):(nu * i)

            block_S[row_range, col_range] = S
        end        

        BTS      = zeros(nu * N, nu * N)
        STB      = zeros(nu * N, nu * N)
        s0T_AT_S = zeros(1, nu * N)

        LinearAlgebra.mul!(BTS, transpose(block_B), block_S)
        LinearAlgebra.mul!(STB, transpose(block_S), block_B)

        LinearAlgebra.axpy!(1, BTS, BQB)
        LinearAlgebra.axpy!(1, STB, BQB)

        LinearAlgebra.mul!(s0T_AT_S, s0TAT, block_S)
        LinearAlgebra.axpy!(1, s0T_AT_S, h)
    end
  
    return (H = BQB, c = vec(h), c0 = c0, block_A = block_A, block_B = block_B, block_Q = block_Q, block_R = block_R)
end

function _build_condensed_G(block_A, block_B, block_E, block_F, block_gl, block_gu, s0, N)
  
    G = zeros(size(block_F))
  
    As0  = zeros(size(block_A, 1), 1)
    EAs0 = zeros(size(block_E, 1), 1)

    LinearAlgebra.mul!(G, block_E, block_B)
    LinearAlgebra.axpy!(1, block_F, G)

    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(EAs0, block_E, As0)
    LinearAlgebra.axpy!(-1, EAs0, block_gl)
    LinearAlgebra.axpy!(-1, EAs0, block_gu)
  
    return (J = G, lcon = vec(block_gl), ucon = vec(block_gu), As0 = As0)
end

for field in fieldnames(LQDynamicData)
    method = Symbol("get_", field)
    @eval begin
        @doc """
            $($method)(LQDynamicData)
            $($method)(LQDynamicModel)
        Return the value $($(QuoteNode(field))) from LQDynamicData or LQDynamicModel.dynamic_data
        """
        $method(dyn_data::LQDynamicData) = getproperty(dyn_data, $(QuoteNode(field)))
    end
    @eval $method(dyn_model::LQDynamicModel) = $method(dyn_model.dynamic_data)
    @eval export $method
end

for field in [:A, :B, :Q, :R, :Qf]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, row, col, val)
            $($method)(LQDynamicModel, row, col, val)
        Set the value of entry $($(QuoteNode(field)))[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data 
        """
        $method(dyn_data::LQDynamicData, row, col, val) = (dyn_data.$field[row, col] = val)
    end
    @eval $method(dyn_model::LQDynamicModel, row, col, val) = (dyn_model.dynamic_data.$field[row,col]=val)
    @eval export $method
end

for field in [:s0, :sl, :su, :ul, :uu]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, index, val)
            $($method)(LQDynamicModel, index, val)
        Set the value of entry $($(QuoteNode(field)))[index] to val for LQDynamicData or LQDynamicModel.dynamic_data 
        """
        $method(dyn_data::LQDynamicData, index, val) = (dyn_data.$field[index] = val)
    end
    @eval $method(dyn_model::LQDynamicModel, index, val) = (dyn_model.dynamic_data.$field[index]=val)
    @eval export $method
end



function fill_structure!(S::SparseMatrixCSC, rows, cols)
  count = 1
  @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
    rows[count] = S.rowval[k]
    cols[count] = col
    count += 1
  end
end

function fill_coord!(S::SparseMatrixCSC, vals, obj_weight)
  count = 1
  @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
    vals[count] = obj_weight * S.nzval[k]
    count += 1
  end
end

function NLPModels.hess_structure!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
  fill_structure!(qp.data.H, rows, cols)
  return rows, cols
end

function NLPModels.hess_structure!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, V, M1 <: Matrix, M2<: Matrix, M3<: Matrix}
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = j:(qp.meta.nvar)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end


function NLPModels.hess_coord!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: Matrix}
  NLPModels.increment!(qp, :neval_hess)
  fill_coord!(qp.data.H, vals, obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, V, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
  NLPModels.increment!(qp, :neval_hess)
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = j:(qp.meta.nvar)
      vals[count] = obj_weight * qp.data.H[i, j]
      count += 1
    end
  end
  return vals
end

NLPModels.hess_coord!(
  qp::LQDynamicModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = NLPModels.hess_coord!(qp, x, vals, obj_weight = obj_weight)

function NLPModels.jac_structure!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
  fill_structure!(qp.data.A, rows, cols)
  return rows, cols
end

function NLPModels.jac_structure!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, V, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = 1:(qp.meta.ncon)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

function NLPModels.jac_coord!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
  NLPModels.increment!(qp, :neval_jac)
  fill_coord!(qp.data.A, vals, one(T))
  return vals
end

function NLPModels.jac_coord!(
  qp::LQDynamicModel{T, V, M1, M2, M3},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, V, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
  NLPModels.increment!(qp, :neval_jac)
  count = 1
  for j = 1:(qp.meta.nvar)
    for i = 1:(qp.meta.ncon)
      vals[count] = qp.data.A[i, j]
      count += 1
    end
  end
  return vals
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
    Q::M, R::M, N;
    Qf::M = Q,
    S::M = zeros(size(Q, 1), size(R, 1))) where M <: AbstractMatrix
    ns = size(Q, 1)
    nu = size(R, 1)

    H = SparseArrays.sparse([],[],eltype(Q)[],(ns * (N + 1) + nu * N), (ns * (N+1) + nu * N))

    for i in 1:N
        range_Q = (1 + (i - 1) * ns): (i * ns)
        range_R = (ns * (N + 1) + 1 + (i - 1) * nu):(ns * (N + 1) + i * nu)
        H[range_Q, range_Q] = Q
        H[range_R, range_R] = R

        H[range_Q, range_R] = S
        H[range_R, range_Q] = transpose(S)
    end

    H[(N * ns + 1):( N * ns + ns), (N * ns + 1):(N * ns + ns)] = Qf

    return H
end




"""
    _build_sparse_J1(A, B, N) -> J

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
function _build_sparse_J1(A::M, B::M, N) where M <: AbstractMatrix

    ns = size(A, 2)
    nu = size(B, 2)

    J1 = SparseArrays.sparse([], [], eltype(A)[], ns * N, (ns* (N + 1) + nu * N))

    neg_ones = .-Matrix(LinearAlgebra.I, ns, ns)

    for i in 1:N
        row_range  = (ns * (i - 1) + 1):(i * ns)
        Acol_range = (ns * (i - 1) + 1):(i * ns)
        Bcol_range = (ns * (N + 1) + 1 + (i - 1) * nu):(ns * (N + 1) + i * nu)
        J1[row_range, Acol_range] = A
        J1[row_range, Bcol_range] = B

        Icol_range = (ns * i + 1):(ns * (i + 1))

        J1[row_range, Icol_range] = neg_ones
    end

    return J1
end

function _build_sparse_J2(E, F, N)
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    J2 = SparseArrays.sparse([],[], eltype(E)[], N * nc, ns * (N + 1) + nu * N)


    if nc != 0
        for i in 1:N
            row_range   = (1 + nc * (i - 1)):(nc * i)
            col_range_E = (1 + ns * (i - 1)):(ns * i)
            col_range_F = (ns * (N + 1) + 1 + nu * (i - 1)):(ns * (N + 1) + nu * i)

            J2[row_range, col_range_E] .= E
            J2[row_range, col_range_F] .= F
        end
    end

    return J2

end

end # module