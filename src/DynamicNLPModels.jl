module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC

export LQDynamicData, SparseLQDynamicModel, DenseLQDynamicModel, get_u, get_s

abstract type AbstractLQDynData{T,V} end
"""
    LQDynamicData{T,V,M,MK} <: AbstractLQDynData{T,V}

A struct to represent the features of the optimization problem 

```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                u_i = Kx_i + v_i  \\forall i = 0, 1, ..., N - 1
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
- `Qf`: objective function matrix for system state at time N
- `S` : objective function matrix for system states and inputs
- `ns`: number of state variables
- `nu`: number of input varaibles
- `E` : constraint matrix for state variables
- `F` : constraint matrix for input variables
- `K` : feedback gain matrix
- `sl`: vector of lower bounds on state variables
- `su`: vector of upper bounds on state variables
- `ul`: vector of lower bounds on input variables
- `uu`: vector of upper bounds on input variables
- `gl`: vector of lower bounds on constraints
- `gu`: vector of upper bounds on constraints

see also `LQDynamicData(s0, A, B, Q, R, N; ...)`
"""
struct LQDynamicData{T, V, M, MK} <: AbstractLQDynData{T,V}
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
    K::MK

    sl::V
    su::V
    ul::V
    uu::V
    gl::V
    gu::V
end

"""
    LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, V, M, MK}
A constructor for building an object of type `LQDynamicData` for the optimization problem 
```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  \\forall i=0, 1, ..., N - 1
                u_i = Kx_i + v_i  \\forall i = 0, 1, ..., N - 1
                gl \\le E s_i + F u_i \\le gu \\forall i = 0, 1, ..., N-1
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
- `S  = nothing`: objective function matrix for system state and inputs
- `E  = zeros(eltype(Q), 0, ns)`  : constraint matrix for state variables
- `F  = zeros(eltype(Q), 0, nu)`  : constraint matrix for input variables
- `K  = nothing`       : feedback gain matrix
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
    S::M  = (similar(Q, size(Q, 1), size(R, 1)) .= 0),
    E::M  = (similar(Q, 0, length(s0)) .= 0),
    F::M  = (similar(Q, 0, size(R, 1)) .= 0),
    K::MK = nothing,

    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf),
    gu::V = (similar(s0, size(F, 1)) .= Inf)
    ) where {T,V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

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
        error("size of Q is not consistent with length of s0")
    end

    if !(sl  <= su)
        error("lower bound(s) on x is > upper bound(s)")
    end
    if !(ul <= uu)
        error("lower bound(s) on u is > upper bound(s)")
    end
    if !(s0 >= sl) || !(s0 <= su)
        error("s0 is not within the given upper and lower bounds")
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
    if K != nothing
        if size(K, 1) != size(R, 1) || size(K, 2) != size(Q,1)
            error("Dimensions of K  do not match number of states and inputs")
        end
    end


    ns = size(Q,1)
    nu = size(R,1)

    LQDynamicData{T, V, M, MK}(
        s0, A, B, Q, R, N,
        Qf, S, ns, nu, E, F, K,
        sl, su, ul, uu, gl, gu
    )
end



abstract type AbstractDynamicModel{T,V} <: QuadraticModels.AbstractQuadraticModel{T, V} end

mutable struct SparseLQDynamicModel{T, V, M1, M2, M3, MK} <:  AbstractDynamicModel{T,V} 
  meta::NLPModels.NLPModelMeta{T, V}
  counters::NLPModels.Counters
  data::QuadraticModels.QPData{T, V, M1, M2}
  dynamic_data::LQDynamicData{T, V, M3, MK}
end

struct Block_Matrices{T, M1, M2}
    A::M1
    B::M1
    Q::M2
    R::M2
    S::M2
    K::M2
    E::M1
    F::M1
    gl::M1
    gu::M1
end

function Block_Matrices(
    A::M1,
    B::M1,
    Q::M2,
    R::M2,
    S::M2,
    K::M2,
    E::M1,
    F::M1,
    gl::M1,
    gu::M1
) where {T, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}}

    Block_Matrices{T, M1, M2}(
        A,
        B,
        Q,
        R,
        S,
        K,
        E,
        F,
        gl,
        gu
    )
end


mutable struct DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK} <:  AbstractDynamicModel{T,V} 
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    data::QuadraticModels.QPData{T, V, M1, M2}
    dynamic_data::LQDynamicData{T, V, M3, MK}
    blocks::Block_Matrices{T, M3, M4}
end

"""
    SparseLQDynamicModel(dnlp::LQDynamicData)    -> SparseLQDynamicModel
    SparseLQDynamicModel(s0, A, B, Q, R, N; ...) -> SparseLQDynamicModel
A constructor for building a `SparseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel`
Input data is for the problem of the form 
```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                u_i = Kx_i + v_i  \\forall i = 0, 1, ..., N - 1
                gl \\le E s_i + F u_i \\le gu for i = 0, 1, ..., N-1            
                sl \\le s \\le su
                ul \\le u \\le uu
                s_0 = s0
```
---

Data is converted to the form 

```math
    minimize    \\frac{1}{2} z^T H z 
    subject to  lcon \\le Jz \\le ucon
                lvar \\le z \\le uvar
```
Resulting `H` and `J` matrices are stored as `QuadraticModels.QPData` within the `SparseLQDynamicModel` struct and 
variable and constraint limits are stored within `NLPModels.NLPModelMeta`

If `K` is defined, then `u` variables are replaced by `v` variables, and `u` can be queried by `get_u` and `get_s` within `DynamicNLPModels.jl`
"""
function SparseLQDynamicModel(dnlp::LQDynamicData{T,V,M}) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}
        _build_sparse_lq_dynamic_model(dnlp)
end

function SparseLQDynamicModel(
    s0::V,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;
    Qf::M = Q, 
    S::M  = (similar(Q, size(Q, 1), size(R, 1)) .= 0),
    E::M  = (similar(Q, 0, length(s0)) .= 0),
    F::M  = (similar(Q, 0, size(R, 1)) .= 0),
    K::MK = nothing,
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf), 
    gu::V = (similar(s0, size(F, 1)) .= Inf), 
    dense=false
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

    dnlp = LQDynamicData(
        s0, A, B, Q, R, N; 
        Qf = Qf, S = S, E = E, F = F, K = K, 
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    
    SparseLQDynamicModel(dnlp)
end

"""
    DenseLQDynamicModel(dnlp::LQDynamicData)    -> DenseLQDynamicModel
    DenseLQDynamicModel(s0, A, B, Q, R, N; ...) -> DenseLQDynamicModel
A constructor for building a `DenseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel`

Input data is for the problem of the form 
```math
    minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
    subject to  s_{i+1} = A s_i + B u_i  for i=0, 1, ..., N-1
                u_i = Kx_i + v_i  \\forall i = 0, 1, ..., N - 1
                gl \\le E s_i + F u_i \\le gu for i = 0, 1, ..., N-1            
                sl \\le s \\le su
                ul \\le u \\le uu
                s_0 = s0
```
---

Data is converted to the form 

```math
    minimize    \\frac{1}{2} u^T H u + h^T u + h0 
    subject to  Jz \\le g
                ul \\le u \\le uu
```

Resulting `H`, `J`, `h`, and `h0` matrices are stored within `QuadraticModels.QPData` as `H`, `A`, `c`, and `c0` attributes respectively

If `K` is defined, then `u` variables are replaced by `v` variables. The bounds on `u` are transformed into algebraic constraints,
and `u` can be queried by `get_u` and `get_s` within `DynamicNLPModels.jl`
"""
function DenseLQDynamicModel(dnlp::LQDynamicData{T,V,M}) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}
    _build_dense_lq_dynamic_model(dnlp)
end

function DenseLQDynamicModel(
    s0::V,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;
    Qf::M = Q, 
    S::M  = (similar(Q, size(Q, 1), size(R, 1)) .= 0),
    E::M  = (similar(Q, 0, length(s0)) .= 0),
    F::M  = (similar(Q, 0, size(R, 1)) .= 0),
    K::MK = nothing,
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf), 
    gu::V = (similar(s0, size(F, 1)) .= Inf), 
    dense=false
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

    dnlp = LQDynamicData(
        s0, A, B, Q, R, N; 
        Qf = Qf, S = S, E = E, F = F, K = K, 
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)
    
    DenseLQDynamicModel(dnlp)
end

function _build_sparse_lq_dynamic_model(dnlp::LQDynamicData{T, V, M, MK}) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: Nothing}
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

    J = vcat(J1, J2)

    c0  = zero(eltype(s0))

    
    nvar = ns * (N + 1) + nu * N
    c  = similar(s0, nvar); fill!(c, 0)
    
    lvar  = similar(s0, nvar); fill!(lvar, 0)
    uvar  = similar(s0, nvar); fill!(uvar, 0)

    lvar[1:ns] .= s0
    uvar[1:ns] .= s0


    lcon  = similar(s0, ns * N + N * length(gl)); fill!(lcon, 0)
    ucon  = similar(s0, ns * N + N * length(gl)); fill!(ucon, 0)

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
    
    SparseLQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        x0   = (similar(s0, nvar) .= 0),
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
        dnlp
    )

end

function _build_sparse_lq_dynamic_model(dnlp::LQDynamicData{T, V, M, MK}) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: AbstractMatrix}
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
    K  = dnlp.K

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu

    # Transform u variables to v variables
    new_Q = similar(Q)
    new_S = similar(S)
    new_A = similar(A)
    new_E = similar(E)

    LinearAlgebra.copyto!(new_Q, Q)
    LinearAlgebra.copyto!(new_S, S)
    LinearAlgebra.copyto!(new_A, A)
    LinearAlgebra.copyto!(new_E, E)

    KTR  = similar(Q, size(K, 2), size(R, 2))
    LinearAlgebra.mul!(KTR, K', R)
    LinearAlgebra.axpy!(1, KTR, new_S)

    SK   = similar(Q, size(S, 1), size(K, 2))
    KTRK = similar(Q, size(K, 2), size(K, 2))
    LinearAlgebra.mul!(SK, S, K)
    LinearAlgebra.mul!(KTRK, KTR, K)
    LinearAlgebra.axpy!(1, SK, new_Q)
    LinearAlgebra.axpy!(1, SK', new_Q)
    LinearAlgebra.axpy!(1, KTRK, new_Q)

    BK    = similar(Q, size(B, 1), size(K, 2))
    LinearAlgebra.mul!(BK, B, K)
    LinearAlgebra.axpy!(1, BK, new_A)

    FK    = similar(Q, size(F, 1), size(K, 2))
    LinearAlgebra.mul!(FK, F, K)
    LinearAlgebra.axpy!(1, FK, new_E)
    
    # Get H and J matrices from new matrices
    H   = _build_H(new_Q, R, N; Qf = Qf, S = new_S)
    J1  = _build_sparse_J1(new_A, B, N)
    J2  = _build_sparse_J2(new_E, F, N)
    J3, lcon3, ucon3  = _build_sparse_J3(K, N, uu, ul)

    J = vcat(J1, J2)
    J = vcat(J, J3)


    nvar = ns * (N + 1) + nu * N
    
    lvar  = similar(s0, nvar); fill!(lvar, -Inf)
    uvar  = similar(s0, nvar); fill!(uvar, Inf)

    lvar[1:ns] .= s0
    uvar[1:ns] .= s0

    lcon  = similar(s0, ns * N + N * length(gl) + length(lcon3)); fill!(lcon, 0)
    ucon  = similar(s0, ns * N + N * length(gl) + length(lcon3)); fill!(ucon, 0)

    ncon  = size(J, 1)
    nnzj = length(J.rowval)
    nnzh = length(H.rowval)

    for i in 1:N
        lvar[(i * ns + 1):((i + 1) * ns)] .= sl
        uvar[(i * ns + 1):((i + 1) * ns)] .= su

        lcon[(ns * N + 1 + (i -1) * length(gl)):(ns * N + i * length(gl))] .= gl
        ucon[(ns * N + 1 + (i -1) * length(gl)):(ns * N + i * length(gl))] .= gu
    end

    if length(lcon3) > 0
        lcon[(1 + ns * N + N * length(gl)):end] .= lcon3
        ucon[(1 + ns * N + N * length(gl)):end] .= ucon3
    end



    c0 = zero(eltype(s0))
    c  = similar(s0, nvar); fill!(c, 0)

    SparseLQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        x0   = (similar(s0, nvar) .= 0),
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
        dnlp
    )
end


function _build_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Nothing}
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
    K  = dnlp.K

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu


    dense_blocks = _build_block_matrices(s0, Q, R, A, B, E, F, N, gu, gl, K; Qf = Qf, S = S)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B
    block_Q  = dense_blocks.Q
    block_R  = dense_blocks.R
    block_E  = dense_blocks.E
    block_F  = dense_blocks.F
    block_S  = dense_blocks.S
    block_K  = dense_blocks.K
    block_gl = dense_blocks.gl
    block_gu = dense_blocks.gu

    H_blocks = _build_H_blocks(block_Q, block_R, block_A, block_B, block_S, block_K, s0, N, K)

    H  = H_blocks.H
    c0 = H_blocks.c0

    G_blocks = _build_G_blocks(block_A, block_B, block_E, block_F, block_K, block_gl, block_gu, s0, N)

    J1    = G_blocks.J
    lcon1 = G_blocks.lcon
    ucon1 = G_blocks.ucon
    As0   = G_blocks.As0

    lvar = similar(s0, nu * N); fill!(lvar, -Inf)
    uvar = similar(s0, nu * N); fill!(uvar, Inf)

    for i in 1:(N)
        lvar[((i - 1) * nu + 1):(i * nu)] = ul
        uvar[((i - 1) * nu + 1):(i * nu)] = uu
    end

    # Convert state variable constraints to algebraic constraints
    bool_vec        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds = sum(bool_vec)

    J2         = similar(Q, num_real_bounds * N, nu * N); fill!(J2, 0)
    As0_bounds = similar(s0, num_real_bounds * N); fill!(As0_bounds, 0)

    if num_real_bounds == length(sl)
        J2        .= block_B[(1 + ns):end,:]
        As0_bounds .= As0[(1 + ns):end,1]
    else        
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds):(i * num_real_bounds)
            J2[row_range, :] .= block_B[(1 + ns * i): (ns * (i + 1)), :][bool_vec, :]
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec]
        end

        sl = sl[bool_vec]
        su = su[bool_vec]
    end

    lcon2 = similar(s0, size(J2, 1)); fill!(lcon2, 0)
    ucon2 = similar(s0, size(J2, 1)); fill!(ucon2, 0)


    for i in 1:N
        lcon2[((i - 1) * length(su) + 1):(i * length(su)),1] = sl
        ucon2[((i - 1) * length(su) + 1):(i * length(su)),1] = su
    end

    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)
    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)

    lcon = similar(s0, length(lcon1) + length(lcon2)); fill!(lcon, 0)
    ucon = similar(s0, length(ucon1) + length(ucon2)); fill!(ucon, 0)
    
    lcon[1:length(lcon1)] .= lcon1
    ucon[1:length(ucon1)] .= ucon1

    if length(lcon2) > 0
        lcon[(1 + length(lcon1)):end] .= lcon2
        ucon[(1 + length(ucon1)):end] .= ucon2
    end

    J   = similar(Q, size(J1, 1) + size(J2, 1), size(J1,2)); fill!(J, 0)
    J[1:(size(J1, 1)), :]   .= J1
    if size(J2, 1) > 0
        J[(1 + size(J1, 1)):end, :] .= J2
    end

    nvar = nu * N
    nnzj = size(J, 1) * size(J, 2)
    nnzh = sum(LinearAlgebra.LowerTriangular(H) .!= 0)
    ncon = size(J, 1)

    c = similar(s0, nvar)
    c .= H_blocks.c

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        x0   = (similar(s0, nvar) .= 0),
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
        c0[1,1], 
        c,
        H,
        J
        ),
        dnlp,
        dense_blocks
    )
end

function _build_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}
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
    K  = dnlp.K

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu


    dense_blocks = _build_block_matrices(s0, Q, R, A, B, E, F, N, gu, gl, K; Qf = Qf, S = S)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B
    block_Q  = dense_blocks.Q
    block_R  = dense_blocks.R
    block_E  = dense_blocks.E
    block_F  = dense_blocks.F
    block_S  = dense_blocks.S
    block_K  = dense_blocks.K
    block_gl = dense_blocks.gl
    block_gu = dense_blocks.gu

    H_blocks = _build_H_blocks(block_Q, block_R, block_A, block_B, block_S, block_K, s0, N, K)

    H  = H_blocks.H
    c0 = H_blocks.c0

    G_blocks = _build_G_blocks(block_A, block_B, block_E, block_F, block_K, block_gl, block_gu, s0, N)

    J1    = G_blocks.J
    lcon1 = G_blocks.lcon
    ucon1 = G_blocks.ucon
    As0   = G_blocks.As0

    # Convert state variable constraints to algebraic constraints
    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds   = sum(bool_vec_s)

    J2         = similar(Q, num_real_bounds * N, nu * N); fill!(J2, 0)
    As0_bounds = similar(s0, num_real_bounds * N); fill!(As0_bounds, 0)


    if num_real_bounds == length(sl)
        J2         .= block_B[(1 + ns):end,:]
        As0_bounds .= As0[(1 + ns):end,1]
    else        
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds):(i * num_real_bounds)
            J2[row_range, :] .= block_B[(1 + ns * i): (ns * (i + 1)), :][bool_vec_s, :]
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s]
        end

        sl = sl[bool_vec_s]
        su = su[bool_vec_s]
    end

    # Convert bounds on u to algebraic constraints
    bool_vec_u       = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds = sum(bool_vec_u)

    KBI  = similar(Q, nu * N, nu * N)
    KAs0 = similar(s0, nu * N)

    I_mat = similar(Q, nu * N, nu * N); fill!(I_mat, 0)

    for i in 1:(nu * N)
        I_mat[i, i] = 1.0
    end

    LinearAlgebra.mul!(KBI, block_K, block_B)
    LinearAlgebra.axpy!(1, I_mat, KBI)
    LinearAlgebra.mul!(KAs0, block_K, As0)

    J3          = similar(Q, num_real_bounds * N, nu * N); fill!(J3, 0)
    KAs0_bounds = similar(s0, num_real_bounds * N); fill!(KAs0_bounds, 0)

    if num_real_bounds == length(ul)
        J3 .= KBI
        KAs0_bounds .= KAs0
    else
        for i in 1:N
            row_range   = (1 + (i - 1) * num_real_bounds):(i * num_real_bounds)
            J3[row_range, :] .= KBI[(1 + nu * (i - 1)):(nu * i), :][bool_vec_u, :]
            KAs0_bounds[row_range]      .= KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]
        end

        ul = ul[bool_vec_u]
        uu = uu[bool_vec_u]
    end

    lcon2 = similar(s0, size(J2, 1)); fill!(lcon2, 0)
    ucon2 = similar(s0, size(J2, 1)); fill!(ucon2, 0)

    lcon3 = similar(s0, size(J3, 1)); fill!(lcon3, 0)
    ucon3 = similar(s0, size(J3, 1)); fill!(ucon3, 0)

    for i in 1:N
        lcon2[((i - 1) * length(su) + 1):(i * length(su)),1] = sl
        ucon2[((i - 1) * length(su) + 1):(i * length(su)),1] = su

        lcon3[((i - 1) * length(uu) + 1):(i * length(uu)),1] = ul
        ucon3[((i - 1) * length(uu) + 1):(i * length(uu)),1] = uu
    end

    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)
    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)

    LinearAlgebra.axpy!(-1, KAs0_bounds, lcon3)
    LinearAlgebra.axpy!(-1, KAs0_bounds, ucon3)


    J = similar(Q, (size(J1, 1) + size(J2, 1) + size(J3, 1)), size(J1, 2)); fill!(J, 0)
    J[1:size(J1, 1), :] .= J1

    if size(J2, 1) > 0
        J[(size(J1, 1) + 1):(size(J1, 1) + size(J2, 1)), :] .= J2
    end

    if size(J3, 1) >0
        J[(size(J1, 1) + size(J2, 1) + 1):(size(J1, 1) + size(J2, 1) + size(J3, 1)), :] .= J3
    end

    lcon = similar(s0, size(J, 1)); fill!(lcon, 0)
    ucon = similar(s0, size(J, 1)); fill!(ucon, 0)

    lcon[1:length(lcon1)] .= lcon1
    ucon[1:length(ucon1)] .= ucon1

    if length(lcon2) > 0
        lcon[(length(lcon1) + 1):(length(lcon1) + length(lcon2))] .= lcon2
        ucon[(length(ucon1) + 1):(length(ucon1) + length(ucon2))] .= ucon2
    end

    if length(lcon3) > 0
        lcon[(length(lcon1) + length(lcon2) + 1):(length(lcon1) + length(lcon2) + length(lcon3))] .= lcon3
        ucon[(length(ucon1) + length(ucon2) + 1):(length(ucon1) + length(ucon2) + length(ucon3))] .= ucon3
    end


    nvar = nu * N
    nnzj = size(J, 1) * size(J, 2)
    nnzh = sum(LinearAlgebra.LowerTriangular(H) .!= 0)
    ncon = size(J, 1)

    c = similar(s0, nvar)
    c .= H_blocks.c

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        x0   = (similar(s0, nvar) .= 0),
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
        c0[1,1], 
        c,
        H,
        J
        ),
        dnlp,
        dense_blocks
    )
end


function _build_block_matrices(
    s0, Q, R, A, B, E, F, N, gu, gl, K;
    Qf = Q, 
    S = (similar(Q, size(Q, 1), size(R, 1)) .= 0)
    )

    ns = size(Q, 1)
    nu = size(R, 1)

    if K == nothing
        K = similar(Q, nu, ns); fill!(K, 0)
    end    
  
    # Define block matrices
    block_B = similar(Q, ns * (N + 1), nu * N); fill!(block_B, 0)
    block_A = similar(Q, ns * (N + 1), ns); fill!(block_A, 0)
    block_Q = SparseArrays.sparse([],[], eltype(Q)[], ns * (N + 1), ns * (N + 1))
    block_R = SparseArrays.sparse([],[], eltype(R)[], nu * N, nu * N)
    block_S = SparseArrays.sparse([],[], eltype(S)[], ns * (N + 1), nu * N)
    block_K = SparseArrays.sparse([],[], eltype(R)[], nu * N, ns * (N + 1))

    nE1 = size(E, 1)
    nE2 = size(E, 2)
    nF1 = size(F, 1)
    nF2 = size(F, 2)
    
    block_E  = similar(Q, nE1 * N, nE2 * (N + 1)); fill!(block_E, 0)
    block_F  = similar(Q, nF1 * N, nF2 * N); fill!(block_F, 0)
    block_gl = similar(Q, nE1 * N, 1); fill!(block_gl, 0)
    block_gu = similar(Q, nE1 * N, 1); fill!(block_gu, 0)
  
    # Build E, F, and d (gl and gu) blocks
    for i in 1:N
        block_E[((i - 1) * nE1 + 1):(i * nE1), ((i - 1) * nE2 + 1):(i * nE2)] = E
        block_F[((i - 1) * nF1 + 1):(i * nF1), ((i - 1) * nF2 + 1):(i * nF2)] = F
        block_gl[((i - 1) * nE1  + 1):(i * nE1)]  = gl
        block_gu[((i - 1) * nE1  + 1):(i * nE1)]  = gu
    end

  
    # Add diagonal of Bs and fill Q, R, S, and K block matrices
    for j in 1:N
        B_row_range = (j * ns + 1):((j + 1) * ns)
        B_col_range = ((j - 1) * nu + 1):(j * nu)
        block_B[B_row_range, B_col_range] = B
  
        block_Q[((j - 1) * ns + 1):(j * ns), ((j - 1) * ns + 1):(j * ns)] = Q
        block_R[((j - 1) * nu + 1):(j * nu), ((j - 1) * nu + 1):(j * nu)] = R

        S_row_range = (1 + ns * (j - 1)):(ns * j)
        S_col_range = (1 + nu * (j - 1)):(nu * j)
        
        block_S[S_row_range, S_col_range] = S

        K_row_range = ((j - 1) * nu + 1):(j * nu)
        block_K[K_row_range, ((j - 1) * ns + 1):(j * ns)] = K
    end

    for i in 1:ns
        block_A[i, i] = 1.0
    end

    A_k = copy(A)
    BK  = similar(Q, size(B, 1), size(K, 2))
    LinearAlgebra.mul!(BK, B, K)
    LinearAlgebra.axpy!(1, BK, A_k)

    # Define matrices for mul!
    A_klast  = copy(A_k)
    A_knext  = copy(A_k)
    AB_klast = similar(Q, size(B, 1), size(B, 2)); fill!(AB_klast, 0)
    AB_k     = similar(Q, size(B, 1), size(B, 2)); fill!(AB_k, 0)
  
    # Fill the A and B matrices
    for i in 1:(N - 1)
        if i == 1
            block_A[(ns + 1):ns*2, :] = A_k
            LinearAlgebra.mul!(AB_k, A_k, B)
            for k in 1:(N-i)
                row_range = (1 + (k + 1) * ns):((k + 2) * ns)
                col_range = (1 + (k - 1) * nu):(k * nu)
                block_B[row_range, col_range] = AB_k
            end
            AB_klast = copy(AB_k)
        else
            LinearAlgebra.mul!(AB_k, A_k, AB_klast)
            LinearAlgebra.mul!(A_knext, A_k, A_klast)
            block_A[(ns * i + 1):ns * (i + 1),:] = A_knext
  
            for k in 1:(N-i)
                row_range = (1 + (k + i) * ns):((k + i + 1) * ns)
                col_range = (1 + (k - 1) * nu):(k * nu)
                block_B[row_range, col_range] = AB_k
            end
  
            AB_klast = copy(AB_k)
            A_klast  = copy(A_knext)
        end
    end

    LinearAlgebra.mul!(A_knext, A_k, A_klast)

    block_A[(ns * N + 1):ns * (N + 1), :] = A_knext
    block_Q[(ns * N + 1):((N + 1) * ns), (N * ns + 1):((N + 1) * ns)] = Qf

    Block_Matrices(
        block_A, 
        block_B, 
        block_Q, 
        block_R,
        block_S,
        block_K,
        block_E,
        block_F,
        block_gl,
        block_gu
    )
end

function _build_H_blocks(block_Q, block_R, block_A::M, block_B::M, block_S, block_K, s0, N, K::MK) where {T, M <: AbstractMatrix{T}, MK <: Nothing}
    As0      = similar(s0, size(block_A, 1))
    QB       = similar(block_A, size(block_Q, 1), size(block_B, 2))
    STB      = similar(block_A, size(block_S, 2), size(block_B, 2))
    B_Q_B    = similar(block_A, size(block_B, 2), size(block_B, 2))

    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(QB, block_Q, block_B)
    LinearAlgebra.mul!(STB, block_S', block_B)
    LinearAlgebra.mul!(B_Q_B, block_B', QB)

    # Define Hessian term so that H = B_Q_B
    LinearAlgebra.axpy!(1, block_R, B_Q_B)
    LinearAlgebra.axpy!(1, STB, B_Q_B)
    LinearAlgebra.axpy!(1, STB', B_Q_B)

    # Define linear term so that c = h
    h = similar(s0, size(block_B, 2))
    LinearAlgebra.axpy!(1, block_S, QB)
    LinearAlgebra.mul!(h, QB', As0)

    # Define linear term so that c0 = h0
    h0   = similar(block_A, 1,1)
    QAs0 = similar(block_A, size(block_Q, 1), 1)
    LinearAlgebra.mul!(QAs0, block_Q, As0)
    LinearAlgebra.mul!(h0, As0', QAs0)

    return (H = B_Q_B, c = h, c0 = h0 ./ T(2))
end

function _build_H_blocks(block_Q, block_R, block_A, block_B, block_S, block_K, s0, N, K::MK) where {T, MK <: AbstractMatrix{T}}

    As0      = similar(s0, size(block_A, 1))
    RK       = similar(block_A, size(block_R, 1), size(block_K, 2))
    RKB      = similar(block_A, size(block_R, 1), size(block_B, 2))
    SK       = similar(block_A, size(block_S, 1), size(block_K, 2))
    SKB      = similar(block_A, size(block_S, 1), size(block_B, 2))
    STB      = similar(block_A, size(block_S, 2), size(block_B, 2))
    KTSTB    = similar(block_A, size(block_K, 2), size(block_B, 2))
    KTRK     = similar(block_A, size(block_K, 2), size(block_K, 2))
    QB       = similar(block_A, size(block_Q, 1), size(block_B, 2))
    KTRK_B   = similar(block_A, size(block_K, 2), size(block_B, 2))
    B_Q_B    = similar(block_A, size(block_B, 2), size(block_B, 2))

    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(RK, block_R, block_K)
    LinearAlgebra.mul!(RKB, RK, block_B)
    LinearAlgebra.mul!(SK, block_S, block_K)
    LinearAlgebra.mul!(SKB, SK, block_B)
    LinearAlgebra.mul!(STB, block_S', block_B)
    LinearAlgebra.mul!(KTSTB, block_K', STB)
    LinearAlgebra.mul!(QB, block_Q, block_B)
    LinearAlgebra.mul!(KTRK, block_K', RK)
    LinearAlgebra.mul!(KTRK_B, KTRK, block_B)

    LinearAlgebra.axpy!(1, KTRK_B, QB)
    LinearAlgebra.axpy!(1, SKB, QB)
    LinearAlgebra.axpy!(1, KTSTB, QB)  # QB now equals QB + KTRKB + SKB + KTSTB

    # Define Hessian term so that H = B_Q_B
    LinearAlgebra.mul!(B_Q_B, block_B', QB)
    LinearAlgebra.axpy!(1, block_R, B_Q_B)
    LinearAlgebra.axpy!(1, RKB', B_Q_B)
    LinearAlgebra.axpy!(1, RKB, B_Q_B)
    LinearAlgebra.axpy!(1, STB, B_Q_B)
    LinearAlgebra.axpy!(1, STB', B_Q_B)

    # Define linear term so that c = h
    h = similar(s0, size(block_B, 2))
    LinearAlgebra.axpy!(1, block_S, QB)
    LinearAlgebra.axpy!(1, RK', QB)
    LinearAlgebra.mul!(h, QB', As0)

    # Define constant term sot hat c0 = h0
    hR_term = similar(block_A, 1, 1) # = s0^T A^T K^T R K A s0
    hS_term = similar(block_A, 1, 1) # = s0^T A^T K^T S^T A s0 = s0^T A^T S K A s0
    hQ_term = similar(block_A, 1, 1) # = s0^T A^T Q A s0

    KTRKAs0 = similar(block_A, size(block_K, 2), 1)
    SKAs0   = similar(block_A, size(block_S, 1), 1)
    QAs0    = similar(block_A, size(block_Q, 1), 1)

    LinearAlgebra.mul!(KTRKAs0, KTRK, As0)
    LinearAlgebra.mul!(SKAs0, SK, As0)
    LinearAlgebra.mul!(QAs0, block_Q, As0)

    LinearAlgebra.mul!(hR_term, As0', KTRKAs0)
    LinearAlgebra.mul!(hS_term, As0', SKAs0)
    LinearAlgebra.mul!(hQ_term, As0', QAs0)
    
    h0 = hR_term ./ T(2) .+ hQ_term ./ T(2) .+ hS_term

    return (H = B_Q_B, c = h, c0 = h0)
end



function _build_G_blocks(block_A, block_B, block_E, block_F, block_K, block_gl, block_gu, s0, N)
  
    G = similar(block_A, size(block_F, 1), size(block_F, 2))
  
    As0  = similar(s0, size(block_A, 1))
    EAs0 = similar(s0, size(block_E, 1))
    FK   = similar(block_A, size(block_F, 1), size(block_K, 2))

    block_E_mul = similar(block_E)
    LinearAlgebra.copyto!(block_E_mul, block_E)

    LinearAlgebra.mul!(FK, block_F, block_K)
    LinearAlgebra.axpy!(1, FK, block_E_mul)
    LinearAlgebra.mul!(G, block_E_mul, block_B)
    LinearAlgebra.axpy!(1, block_F, G)

    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(EAs0, block_E_mul, As0)
    LinearAlgebra.axpy!(-1, EAs0, block_gl)
    LinearAlgebra.axpy!(-1, EAs0, block_gu)
  
    return (J = G, lcon = vec(block_gl), ucon = vec(block_gu), As0 = As0)
end

"""
    get_u(solution_ref, lqdm::SparseLQDynamicModel) -> u <: vector
    get_u(solution_ref, lqdm::DenseLQDynamicModel) -> u <: vector

Query the solution `u` from the solver. If `K = nothing`, the solution for `u` is queried from `solution_ref.solution`

If `K <: AbstractMatrix`, `solution_ref.solution` returns `v`, and `get_u` solves for `u` using the `K` matrix (and the `A` and `B` matrices if `lqdm <: DenseLQDynamicModel`)
"""
function get_u(
    solver_status, 
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}

    solution = solver_status.solution
    ns       = lqdm.dynamic_data.ns
    nu       = lqdm.dynamic_data.nu
    N        = lqdm.dynamic_data.N
    K        = lqdm.dynamic_data.K

    u = zeros(T, nu * N)

    for i in 1:N
        start_v = (i - 1) * nu + 1
        end_v   = i * nu
        start_s = (i - 1) * ns + 1
        end_s   = i * ns

        Ks = zeros(T, size(K, 1), 1)

        s = solution[start_s:end_s]
        v = solution[(ns * (N + 1) + start_v):(ns * (N + 1) + end_v)]

        LinearAlgebra.mul!(Ks, K, s)
        LinearAlgebra.axpy!(1, v, Ks)

        u[start_v:end_v] = Ks
    end
    return u
end

function get_u(
    solver_status, 
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, M4 <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}

    dnlp = lqdm.dynamic_data
    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B
    block_K = lqdm.blocks.K

    v = solver_status.solution

    As0 = zeros(T, size(block_A, 1))
    Bv  = zeros(T, size(block_B, 1))

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.mul!(Bv, block_B, v)
    LinearAlgebra.axpy!(1, As0, Bv)

    Ks = zeros(T, size(block_K, 1))

    LinearAlgebra.mul!(Ks, block_K, Bv)

    u = copy(v)
    LinearAlgebra.axpy!(1, Ks, u)

    return vec(u)
end

function get_u(
    solver_status, 
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, MK <: Nothing}

    solution = solver_status.solution
    ns       = lqdm.dynamic_data.ns
    nu       = lqdm.dynamic_data.nu
    N        = lqdm.dynamic_data.N

    u = solution[(ns * (N + 1) + 1):end]
    return u
end

function get_u(
    solver_status, 
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, M4 <: AbstractMatrix{T}, MK <: Nothing}

    return copy(solver_status.solution)
end

"""
    get_s(solution_ref, lqdm::SparseLQDynamicModel) -> s <: vector
    get_s(solution_ref, lqdm::DenseLQDynamicModel) -> s <: vector

Query the solution `s` from the solver. If `lqdm <: SparseLQDynamicModel`, the solution is queried directly from `solution_ref.solution`
If `lqdm <: DenseLQDynamicModel`, then `solution_ref.solution` returns `u` (if `K = nothing`) or `v` (if `K <: AbstactMatrix`), and `s` is found form
transforming `u` or `v` into `s` using `A`, `B`, and `K` matrices.
"""
function get_s(
    solver_status, 
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix}}

    solution = solver_status.solution
    ns       = lqdm.dynamic_data.ns
    N        = lqdm.dynamic_data.N

    s = solution[1:(ns * (N + 1))]
    return s
end
function get_s(
    solver_status,
    lqdm::DenseLQDynamicModel{T,V, M1, M2, M3, M4, MK}
    ) where {T, V <: AbstractVector{T}, M1 <: AbstractMatrix{T}, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}, M4 <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix}}

    dnlp = lqdm.dynamic_data
    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B

    v = solver_status.solution

    As0 = zeros(T, size(block_A, 1))
    Bv  = zeros(T, size(block_B, 1))

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.mul!(Bv, block_B, v)
    LinearAlgebra.axpy!(1, As0, Bv)

    return vec(Bv)
end


for field in fieldnames(LQDynamicData)
    method = Symbol("get_", field)
    @eval begin
        @doc """
            $($method)(LQDynamicData)
            $($method)(SparseLQDynamicModel)
            $($method)(DenseLQDynamicModel)
        Return the value $($(QuoteNode(field))) from LQDynamicData or SparseLQDynamicModel.dynamic_data or DenseLQDynamicModel.dynamic_data
        """
        $method(dyn_data::LQDynamicData) = getproperty(dyn_data, $(QuoteNode(field)))
    end
    @eval $method(dyn_model::SparseLQDynamicModel) = $method(dyn_model.dynamic_data)
    @eval $method(dyn_model::DenseLQDynamicModel)  = $method(dyn_model.dynamic_data)
    @eval export $method
end

for field in [:A, :B, :Q, :R, :Qf, :E, :F, :S, :K]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, row, col, val)
            $($method)(SparseLQDynamicModel, row, col, val)
            $($method)(DenseLQDynamicModel, row, col, val)
        Set the value of entry $($(QuoteNode(field)))[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamic_data, or DenseLQDynamicModel.dynamic_data
        """
        $method(dyn_data::LQDynamicData, row, col, val) = (dyn_data.$field[row, col] = val)
    end
    @eval $method(dyn_model::SparseLQDynamicModel, row, col, val) = (dyn_model.dynamic_data.$field[row, col] = val)
    @eval $method(dyn_model::DenseLQDynamicModel, row, col, val)  = (dyn_model.dynamic_data.$field[row, col] = val)
    @eval export $method
end

for field in [:s0, :sl, :su, :ul, :uu, :gl, :gu]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, index, val)
            $($method)(SparseLQDynamicModel, index, val)
            $($method)(DenseLQDynamicModel, index, val)
        Set the value of entry $($(QuoteNode(field)))[index] to val for LQDynamicData, SparseLQDynamicModel.dynamic_data, or DenseLQDynamicModel.dynamic_data
        """
        $method(dyn_data::LQDynamicData, index, val) = (dyn_data.$field[index] = val)
    end
    @eval $method(dyn_model::SparseLQDynamicModel, index, val) = (dyn_model.dynamic_data.$field[index] = val)
    @eval $method(dyn_model::DenseLQDynamicModel, index, val)  = (dyn_model.dynamic_data.$field[index] = val)
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
    qp::SparseLQDynamicModel{T, V, M1, M2, M3},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
    fill_structure!(qp.data.H, rows, cols)
    return rows, cols
end

  
function NLPModels.hess_structure!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
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
    qp::SparseLQDynamicModel{T, V, M1, M2, M3},
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight::Real = one(eltype(x)),
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: Matrix}
    NLPModels.increment!(qp, :neval_hess)
    fill_coord!(qp.data.H, vals, obj_weight)
    return vals
end

function NLPModels.hess_coord!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
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
    qp::SparseLQDynamicModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real = one(eltype(x)),
) = NLPModels.hess_coord!(qp, x, vals, obj_weight = obj_weight)

NLPModels.hess_coord!(
    qp::DenseLQDynamicModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real = one(eltype(x)),
) = NLPModels.hess_coord!(qp, x, vals, obj_weight = obj_weight)

function NLPModels.jac_structure!(
    qp::SparseLQDynamicModel{T, V, M1, M2, M3},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3<: AbstractMatrix}
    fill_structure!(qp.data.A, rows, cols)
    return rows, cols
end

function NLPModels.jac_structure!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
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
    qp::SparseLQDynamicModel{T, V, M1, M2, M3},
    x::AbstractVector,
    vals::AbstractVector,
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
    NLPModels.increment!(qp, :neval_jac)
    fill_coord!(qp.data.A, vals, one(T))
    return vals
end

function NLPModels.jac_coord!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
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
    S::Union{M, Nothing} = nothing) where M <: AbstractMatrix
    ns = size(Q, 1)
    nu = size(R, 1)

    H = SparseArrays.sparse([],[],eltype(Q)[],(ns * (N + 1) + nu * N), (ns * (N+1) + nu * N))

    for i in 1:N
        range_Q = (1 + (i - 1) * ns): (i * ns)
        range_R = (ns * (N + 1) + 1 + (i - 1) * nu):(ns * (N + 1) + i * nu)
        H[range_Q, range_Q] = Q
        H[range_R, range_R] = R
        if S != nothing
            H[range_Q, range_R] = S
            H[range_R, range_Q] = S'
        end
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

            J2[row_range, col_range_E] = E
            J2[row_range, col_range_F] = F
        end
    end

    return J2

end

function _build_sparse_J3(K, N, uu, ul)

    # Remove algebraic constraints if u variable is unbounded on both upper and lower ends
    nu = length(ul)
    ns = size(K, 2)

    bool_vec        = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds = sum(bool_vec)

    J3 = SparseArrays.sparse([],[],eltype(K)[], nu * N, ns * (N + 1) + nu * N)
    I_mat = Matrix(LinearAlgebra.I, nu, nu)

    full_bool_vec = fill(true, nu * N)

    lcon3 = similar(ul, nu * N); fill!(lcon3, 0)
    ucon3 = similar(ul, nu * N); fill!(ucon3, 0)

    for i in 1:N
        row_range   = (nu * (i - 1) + 1):(nu * i)
        K_col_range = (ns * (i - 1) + 1):(ns * i)
        I_col_range = (ns * (N + 1) + 1 + nu * (i - 1)):(ns * (N + 1) + nu * i)
        J3[row_range, K_col_range] = K
        J3[row_range, I_col_range] = I_mat

        lcon3[row_range] .= ul
        ucon3[row_range] .= uu
        full_bool_vec[row_range] = bool_vec
    end

    return J3[full_bool_vec, :], lcon3[full_bool_vec], ucon3[full_bool_vec]
end

end # module