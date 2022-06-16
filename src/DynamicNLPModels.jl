module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC

export get_QM, LQDynamicData, LQDynamicModel

abstract type AbstractLQDynData{T,S} end
"""
    LQDynamicData{T,S,M} <: AbstractLQDynData{T,S}

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
struct LQDynamicData{T,S,M} <: AbstractLQDynData{T,S}
    s0::S
    A::M
    B::M
    Q::M
    R::M
    N

    Qf::M
    ns::Int
    nu::Int
    E::M
    F::M

    sl::S
    su::S
    ul::S
    uu::S
    gl::S
    gu::S
end

"""
    LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, S, M}
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
    s0::S,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;

    Qf::M = Q, 
    E::M  = zeros(0, length(s0)),
    F::M  = zeros(0, size(R, 1)),

    sl::S = (similar(s0) .= -Inf),
    su::S = (similar(s0) .=  Inf),
    ul::S = (similar(s0,size(R,1)) .= -Inf),
    uu::S = (similar(s0,size(R,1)) .=  Inf),
    gl::S = fill(-Inf, size(E, 1)),
    gu::S = fill(Inf, size(F, 1))
    ) where {T,S <: AbstractVector{T},M <: AbstractMatrix{T}}

    if size(Q,1) != size(Q,2) 
        error("Q matrix is not square")
    end
    if size(R,1) != size(R,1)
        error("R matrix is not square")
    end
    if size(A,2) != length(s0)
        error("Number of columns of A are not equal to the number of states")
    end
    if size(B,2) != size(R,1)
        error("Number of columns of B are not equal to the number of inputs")
    end
    if length(s0) != size(Q,1)
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

    if size(E,1) != size(F,1)
        error("E and F have different numbers of rows")
    end
    if !(gl <= gu)
        error("lower bound(s) on Es + Fu is > upper bound(s)")
    end
    if size(E,2) != size(Q, 1)
        error("Dimensions of E are not the same as number of states")
    end
    if size(F,2) != size(R,1) 
        error("Dimensions of F are not the same as the number of inputs")
    end
    if length(gl) != size(E,1)
        error("Dimensions of gl do not match E and F")
    end
    if length(gu) != size(E,1)
        error("Dimensions of gu do not match E and F")
    end



    ns= size(Q,1)
    nu= size(R,1)

    LQDynamicData{T,S,M}(
        s0, A, B, Q, R, N,
        Qf, ns, nu, E, F,
        sl, su, ul, uu, gl, gu
    )
end



abstract type AbstractDynamicModel{T,S} <: QuadraticModels.AbstractQuadraticModel{T, S} end

mutable struct LQDynamicModel{T, S, M1, M2, M3} <:  AbstractDynamicModel{T,S} 
  meta::NLPModels.NLPModelMeta{T, S}
  counters::NLPModels.Counters
  data::QuadraticModels.QPData{T, S, M1, M2}
  dynamic_data::LQDynamicData{T,S,M3}
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
function LQDynamicModel(dnlp::LQDynamicData{T,S,M}; condense = false) where {T,S <: AbstractVector{T} ,M  <: AbstractMatrix{T}}
    s0 = dnlp.s0
    A  = dnlp.A
    B  = dnlp.B
    Q  = dnlp.Q
    R  = dnlp.R
    N  = dnlp.N

    Qf = dnlp.Qf
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


    if condense == false
        H  = _build_H(Q, R, N; Qf=Qf)
        J  = _build_J(A, B, N)
        EF = _build_EF(E, F, N)

        c0 = 0.0

        nvar = ns * (N + 1) + nu * N

        c  = zeros(nvar)

        lvar  = copy(s0)
        uvar  = copy(s0)
        ucon  = zeros(size(J, 1))
        lcon  = zeros(size(J, 1))
        J     = vcat(J, EF)
        ncon  = size(J, 1)

        nnzj = length(J.rowval)
        nnzh = length(H.rowval)


        for i in 1:N
            lvar = vcat(lvar, sl)
            uvar = vcat(uvar, su)
            lcon = vcat(lcon, gl)
        end

        for j in 1:N
            lvar = vcat(lvar, ul)
            uvar = vcat(uvar, uu)
            ucon = vcat(ucon, gu)
        end
    else
        condensed_blocks = _build_condensed_blocks(s0, Q, R, A, B, N; Qf = Qf)

        block_A = condensed_blocks.block_A
        block_B = condensed_blocks.block_B
        H       = condensed_blocks.H
        c       = condensed_blocks.c
        c0      = condensed_blocks.c0

        lvar = copy(ul)
        uvar = copy(uu)

        for i in 1:(N-1)
            lvar = vcat(lvar, ul)
            uvar = vcat(uvar, uu)
        end

        d    = fill(Inf, ns * 2)

        d[1:ns] .= .-sl
        d[(ns+1):(2*ns)] .= su


        E_bounds = zeros(ns * 2, ns)
        F_bounds = zeros(ns * 2, nu)

        for i in 1:ns
            E_bounds[i,i] = -1.0
            E_bounds[i + ns, i] = 1.0
        end

        E_bounds = vcat(E_bounds, E)
        F_bounds = vcat(F_bounds, F)
        d        = vcat(d, gu)

        E = vcat(E_bounds, .-E)
        F = vcat(F_bounds, .-F)
        d = vcat(d, .-gl)
        
        J, ucon = _build_G(block_A, block_B, E, F, d, s0, N)

        lcon = fill(-Inf, length(ucon))

        nvar = nu * N
        nnzj = size(J, 1) * size(J, 2)
        nnzh = sum(LinearAlgebra.LowerTriangular(H) .!= 0)
        ncon = size(J, 1)
        
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
        condense
    )

end


function LQDynamicModel(
    s0::S,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;
    Qf::M = Q, 
    E::M  = zeros(0, length(s0)),
    F::M  = zeros(0, size(R, 1)),
    sl::S = (similar(s0) .= -Inf),
    su::S = (similar(s0) .=  Inf),
    ul::S = (similar(s0,size(R, 1)) .= -Inf),
    uu::S = (similar(s0,size(R, 1)) .=  Inf),
    gl::S = fill(-Inf, size(E, 1)),
    gu::S = fill(Inf, size(F, 1)),
    condense=false
    ) where {T,S <: AbstractVector{T},M <: AbstractMatrix{T}}

    dnlp = LQDynamicData(s0, A, B, Q, R, N; Qf = Qf, E = E, F = F, sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    
    LQDynamicModel(dnlp; condense=condense)

end


function _build_condensed_blocks(
    s0, Q, R, A, B, N;
    Qf = Q)
  
    ns = size(Q, 1)
    nu = size(R, 1)
  
    # Define block matrices
    block_B = zeros(ns * (N + 1), nu * N)
    block_A = zeros(ns * (N + 1), ns)
    block_Q = SparseArrays.sparse([],[], Float64[], ns * (N + 1), ns * (N + 1))
    block_R = SparseArrays.sparse([],[], Float64[], nu * N, nu * N)
  
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
  
  
    return (H = BQB, c = vec(h), c0 = c0, block_A = block_A, block_B = block_B, block_Q = block_Q, block_R = block_R)
end

function _build_G(block_A, block_B, E, F, d,s0, N)
    
    nE1 = size(E, 1)
    nE2 = size(E, 2)
    nF1 = size(F, 1)
    nF2 = size(F, 2)
    nd  = length(d)
    
    block_E = zeros(nE1 * N, nE2 * (N + 1))
    block_F = zeros(nF1 * N, nF2 * N)
    block_d = zeros(nd * N, 1)
  
    for i in 1:N
        block_E[((i - 1) * nE1 + 1):(i * nE1), ((i - 1) * nE2 + 1):(i * nE2)] .= E
        block_F[((i - 1) * nF1 + 1):(i * nF1), ((i - 1) * nF2 + 1):(i * nF2)] .= F
        block_d[((i - 1) * nd  + 1):(i * nd)]  .= d
    end
  
    G = zeros(size(block_F))
  
    As0  = zeros(size(block_A, 1), 1)
    EAs0 = zeros(size(block_E, 1), 1)

    LinearAlgebra.axpy!(1, block_F, LinearAlgebra.mul!(G, block_E, block_B))
    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(EAs0, block_E, As0)
    LinearAlgebra.axpy!(-1, EAs0, block_d)
  
    return G, vec(block_d)
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
  qp::LQDynamicModel{T, S, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
  fill_structure!(qp.data.H, rows, cols)
  return rows, cols
end

function NLPModels.hess_structure!(
  qp::LQDynamicModel{T, S, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: Matrix, M2<: Matrix, M3<: Matrix}
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
  qp::LQDynamicModel{T, S, M1, M2, M3},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: Matrix}
  NLPModels.increment!(qp, :neval_hess)
  fill_coord!(qp.data.H, vals, obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  qp::LQDynamicModel{T, S, M1, M2, M3},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
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
  qp::LQDynamicModel{T, S, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
  fill_structure!(qp.data.A, rows, cols)
  return rows, cols
end

function NLPModels.jac_structure!(
  qp::LQDynamicModel{T, S, M1, M2, M3},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1<: Matrix, M2 <: Matrix, M3 <: Matrix}
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
  qp::LQDynamicModel{T, S, M1, M2, M3},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
  NLPModels.increment!(qp, :neval_jac)
  fill_coord!(qp.data.A, vals, one(T))
  return vals
end

function NLPModels.jac_coord!(
  qp::LQDynamicModel{T, S, M1, M2, M3},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
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
    Q, R, N;
    Qf = [])
    if size(Qf,1) == 0
        Qf = copy(Q)
    end

    ns = size(Q, 1)
    nu = size(R, 1)

    H = SparseArrays.sparse([],[],Float64[],(ns * (N + 1) + nu * N), (ns * (N+1) + nu * N))

    for i in 1:N
        for j in 1:ns
            for k in 1:ns
                row_index = (i - 1) * ns + k
                col_index = (i - 1) * ns + j
                H[row_index, col_index] = Q[k,j]

            end
        end
    end

    for j in 1:ns
        for k in 1:ns
            row_index = N * ns + k
            col_index = N * ns + j
            H[row_index, col_index] = Qf[k,j]
        end
    end


    for i in 1:N
        for j in 1:nu
            for k in 1:nu
                row_index = ns * (N + 1) + (i - 1) * nu + k
                col_index = ns * (N + 1) + (i - 1) * nu + j
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
    ns = size(A, 2)
    nu = size(B, 2)


    J = SparseArrays.sparse([], [], Float64[], ns * N, (ns* (N + 1) + nu * N))    

    for i in 1:N
        for j in 1:ns
            row_index = (i - 1) * ns + j
            J[row_index, (i * ns + j)] = -1
            for k in 1:ns
                col_index = (i - 1) * ns + k
                J[row_index, col_index] = A[j,k]
            end

            for k in 1:nu
                col_index = ((N + 1) * ns) + (i - 1) * nu + k
                J[row_index, col_index] = B[j,k]    
            end
        end
    end

    return J
end

function _build_EF(E, F, N)
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    EF = SparseArrays.sparse([],[], Float64[], N * nc, ns * (N + 1) + nu * N)


    if nc != 0
        for i in 1:N
            row_range   = (1 + nc * (i - 1)):(nc * i)
            col_range_E = (1 + ns * (i - 1)):(ns * i)
            col_range_F = (ns * (N + 1) + 1 + nu * (i - 1)):(ns * (N + 1) + nu * i)

            EF[row_range, col_range_E] .= E
            EF[row_range, col_range_F] .= F
        end
    end

    return EF

end

end # module

 