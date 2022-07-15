module DynamicNLPModels

import NLPModels
import QuadraticModels
import LinearAlgebra
import SparseArrays
import LinearOperators

import SparseArrays: SparseMatrixCSC

export LQDynamicData, SparseLQDynamicModel, DenseLQDynamicModel, get_u, get_s, get_jacobian, add_jtsj!

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
    N::Int

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
    S::M  = _init_similar(Q, size(Q, 1), size(R, 1), T),
    E::M  = _init_similar(Q, 0, length(s0), T),
    F::M  = _init_similar(Q, 0, size(R, 1), T),
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

    if _cmp_arr(<, sl, su)
        error("lower bound(s) on s is > upper bound(s)")
    end
    if _cmp_arr(<, ul, uu)
        error("lower bound(s) on u is > upper bound(s)")
    end
    if _cmp_arr(>=, s0, sl) || _cmp_arr(<=, s0, su)
        error("s0 is not within the given upper and lower bounds")
    end

    if size(E, 1) != size(F, 1)
        error("E and F have different numbers of rows")
    end
    if _cmp_arr(<, gl, gu)
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

struct SparseLQDynamicModel{T, V, M1, M2, M3, MK} <:  AbstractDynamicModel{T,V}
  meta::NLPModels.NLPModelMeta{T, V}
  counters::NLPModels.Counters
  data::QuadraticModels.QPData{T, V, M1, M2}
  dynamic_data::LQDynamicData{T, V, M3, MK}
end

"""
Struct containing block A and B matrices used in creating the `DenseLQDynamicModel`. These matrices are given by Jerez, Kerrigan, and Constantinides
in section 4 of "A sparse and condensed QP formulation for predictive control of LTI systems" (doi:10.1016/j.automatica.2012.03.010).

`A` is a ns(N+1) x ns matrix and `B` is a ns(N) x nu matrix containing the first column of the `B` block matrix in the above text. Note that the
first block of zeros is omitted.
"""
struct DenseLQDynamicBlocks{T, M}
    A::M
    B::M
end

struct DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK} <:  AbstractDynamicModel{T,V}
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    data::QuadraticModels.QPData{T, V, M1, M2}
    dynamic_data::LQDynamicData{T, V, M3, MK}
    blocks::DenseLQDynamicBlocks{T, M4}
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
    S::M  = _init_similar(Q, size(Q, 1), size(R, 1), T),
    E::M  = _init_similar(Q, 0, length(s0), T),
    F::M  = _init_similar(Q, 0, size(R, 1), T),
    K::MK = nothing,
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf),
    gu::V = (similar(s0, size(F, 1)) .= Inf)
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

    dnlp = LQDynamicData(
        s0, A, B, Q, R, N;
        Qf = Qf, S = S, E = E, F = F, K = K,
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)

    SparseLQDynamicModel(dnlp)
end

"""
    DenseLQDynamicModel(dnlp::LQDynamicData; implicit = false)    -> DenseLQDynamicModel
    DenseLQDynamicModel(s0, A, B, Q, R, N; implicit = false ...) -> DenseLQDynamicModel
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

Keyword argument `implicit = false` determines how the Jacobian is stored within the `QPData`. If `implicit = false`, the full, dense
Jacobian matrix is stored. If `implicit = true`, only the first `nu` columns of the Jacobian are stored with the Linear Operator `LQJacobianOperator`.
"""
function DenseLQDynamicModel(dnlp::LQDynamicData{T,V,M}; implicit = false) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}
    if implicit
        _build_implicit_dense_lq_dynamic_model(dnlp)
    else
        _build_dense_lq_dynamic_model(dnlp)
    end
end

function DenseLQDynamicModel(
    s0::V,
    A::M,
    B::M,
    Q::M,
    R::M,
    N;
    Qf::M = Q,
    S::M  = _init_similar(Q, size(Q, 1), size(R, 1), T),
    E::M  = _init_similar(Q, 0, length(s0), T),
    F::M  = _init_similar(Q, 0, size(R, 1), T),
    K::MK = nothing,
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf),
    gu::V = (similar(s0, size(F, 1)) .= Inf),
    implicit = false
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

    dnlp = LQDynamicData(
        s0, A, B, Q, R, N;
        Qf = Qf, S = S, E = E, F = F, K = K,
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu)

    DenseLQDynamicModel(dnlp; implicit = implicit)
end

"""
    LQJacobianOperator{T, V, M}

Struct for storing the implicit Jacobian matrix. All data for the Jacobian can be stored
in the first `nu` columns of `J`. This struct contains the needed data and storage arrays for
calculating `Jx`, `J^T x`, and `J^T Sigma J`. `Jx` and `J^T x` are performed through extensions
to `LinearAlgebra.mul!()`.

---
Attributes
 - `truncated_jac`: Matrix of first `nu` columns of the Jacobian
 - `N`  : number of time steps
 - `nu` : number of inputs
 - `nc` : number of algebraic constraints of the form gl <= Es + Fu <= gu
 - `nsc`: number of bounded state variables
 - `nuc`: number of bounded input variables (if `K` is defined)
 - `SJ1`: placeholder for storing data when calculating `ΣJ`
 - `SJ2`: placeholder for storing data when calculating `ΣJ`
 - `SJ3`: placeholder for storing data when calculating `ΣJ`

"""
struct LQJacobianOperator{T, V, M} <: LinearOperators.AbstractLinearOperator{T}
    truncated_jac::M  # column of Jacobian block matrix
    N::Int            # number of time steps
    nu::Int           # number of inputs
    nc::Int           # number of inequality constraints
    nsc::Int          # number of state variables that are constrained
    nuc::Int          # number of input variables that are constrained

    # Storage matices for building J^TΣJ
    SJ1::M
    SJ2::M
    SJ3::M
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
    K  = dnlp.K

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu

    nc = size(E, 1)

    H_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    H_rowval = zeros(Int, (ns + nu) * N * ns + (ns + nu) * N * nu + ns * ns)
    H_nzval  = zeros(T, (ns + nu) * N * ns + (ns + nu) * N * nu + ns * ns)

    J_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    J_rowval = zeros(Int, N * (ns^2 + ns * nu + ns) + N * (nc * ns + nc * nu))
    J_nzval  = zeros(T, N * (ns^2 + ns * nu + ns) + N * (nc * ns + nc * nu))

    _set_sparse_H!(H_colptr, H_rowval, H_nzval, Q, R, N; Qf = Qf, S = S)

    H = SparseArrays.SparseMatrixCSC((N + 1) * ns + nu * N, (N + 1) * ns + nu * N, H_colptr, H_rowval, H_nzval)

    _set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, N)

    J = SparseArrays.SparseMatrixCSC((nc + ns) * N, (N + 1) * ns + nu * N, J_colptr, J_rowval, J_nzval)

    SparseArrays.dropzeros!(H)
    SparseArrays.dropzeros!(J)

    c0  = zero(T)

    nvar = ns * (N + 1) + nu * N
    c  = _init_similar(s0, nvar, T)

    lvar  = _init_similar(s0, nvar, T)
    uvar  = _init_similar(s0, nvar, T)

    lvar[1:ns] = s0
    uvar[1:ns] = s0

    lcon  = _init_similar(s0, ns * N + N * nc, T)
    ucon  = _init_similar(s0, ns * N + N * nc, T)

    ncon  = size(J, 1)
    nnzj = length(J.rowval)
    nnzh = length(H.rowval)

    for i in 1:N
        lvar[(i * ns + 1):((i + 1) * ns)] = sl
        uvar[(i * ns + 1):((i + 1) * ns)] = su

        lcon[(ns * N + 1 + (i -1) * nc):(ns * N + i * nc)] = gl
        ucon[(ns * N + 1 + (i -1) * nc):(ns * N + i * nc)] = gu
    end

    for j in 1:N
        lvar[((N + 1) * ns + (j - 1) * nu + 1):((N + 1) * ns + j * nu)] = ul
        uvar[((N + 1) * ns + (j - 1) * nu + 1):((N + 1) * ns + j * nu)] = uu
    end

    SparseLQDynamicModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0   = _init_similar(s0, nvar, T),
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

    nc = size(E, 1)

    bool_vec        = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds = sum(bool_vec)

    # Transform u variables to v variables
    new_Q = _init_similar(Q, size(Q, 1), size(Q, 2), T)
    new_S = _init_similar(S, size(S, 1), size(S, 2), T)
    new_A = _init_similar(A, size(A, 1), size(A, 2), T)
    new_E = _init_similar(E, size(E, 1), size(E, 2), T)
    KTR   = _init_similar(Q, size(K, 2), size(R, 2), T)
    SK    = _init_similar(Q, size(S, 1), size(K, 2), T)
    KTRK  = _init_similar(Q, size(K, 2), size(K, 2), T)
    BK    = _init_similar(Q, size(B, 1), size(K, 2), T)
    FK    = _init_similar(Q, size(F, 1), size(K, 2), T)

    H_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    H_rowval = zeros(Int, (ns + nu) * N * ns + (ns + nu) * N * nu + ns * ns)
    H_nzval  = zeros(T, (ns + nu) * N * ns + (ns + nu) * N * nu + ns * ns)

    J_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    J_rowval = zeros(Int, N * (ns^2 + ns * nu + ns) + N * (nc * ns + nc * nu) + N * (ns * num_real_bounds + num_real_bounds))
    J_nzval  = zeros(T, N * (ns^2 + ns * nu + ns) + N * (nc * ns + nc * nu) + N * (ns * num_real_bounds + num_real_bounds))

    LinearAlgebra.copyto!(new_Q, Q)
    LinearAlgebra.copyto!(new_S, S)
    LinearAlgebra.copyto!(new_A, A)
    LinearAlgebra.copyto!(new_E, E)

    LinearAlgebra.mul!(KTR, K', R)
    LinearAlgebra.axpy!(1, KTR, new_S)

    LinearAlgebra.mul!(SK, S, K)
    LinearAlgebra.mul!(KTRK, KTR, K)
    LinearAlgebra.axpy!(1, SK, new_Q)
    LinearAlgebra.axpy!(1, SK', new_Q)
    LinearAlgebra.axpy!(1, KTRK, new_Q)

    LinearAlgebra.mul!(BK, B, K)
    LinearAlgebra.axpy!(1, BK, new_A)

    LinearAlgebra.mul!(FK, F, K)
    LinearAlgebra.axpy!(1, FK, new_E)

    # Get H and J matrices from new matrices
    _set_sparse_H!(H_colptr, H_rowval, H_nzval, new_Q, R, N; Qf = Qf, S = new_S)

    H = SparseArrays.SparseMatrixCSC((N + 1) * ns + nu * N, (N + 1) * ns + nu * N, H_colptr, H_rowval, H_nzval)

    _set_sparse_J!(J_colptr, J_rowval, J_nzval, new_A, B, new_E, F, K, bool_vec, N, num_real_bounds)

    J = SparseArrays.SparseMatrixCSC(ns * N + nc * N + num_real_bounds * N, (N + 1) * ns + nu * N, J_colptr, J_rowval, J_nzval)

    SparseArrays.dropzeros!(H)
    SparseArrays.dropzeros!(J)

    # Remove algebraic constraints if u variable is unbounded on both upper and lower ends


    lcon3 = _init_similar(ul, nu * N, T)
    ucon3 = _init_similar(ul, nu * N, T)

    ul = ul[bool_vec]
    uu = uu[bool_vec]

    lcon3 = repeat(ul, N)
    ucon3 = repeat(uu, N)

    nvar = ns * (N + 1) + nu * N

    lvar  = similar(s0, nvar); fill!(lvar, -Inf)
    uvar  = similar(s0, nvar); fill!(uvar, Inf)

    lvar[1:ns] = s0
    uvar[1:ns] = s0

    lcon  = _init_similar(s0, ns * N + N * length(gl) + length(lcon3))
    ucon  = _init_similar(s0, ns * N + N * length(gl) + length(lcon3))

    ncon  = size(J, 1)
    nnzj = length(J.rowval)
    nnzh = length(H.rowval)

    for i in 1:N
        lvar[(i * ns + 1):((i + 1) * ns)] = sl
        uvar[(i * ns + 1):((i + 1) * ns)] = su

        lcon[(ns * N + 1 + (i -1) * nc):(ns * N + i * nc)] = gl
        ucon[(ns * N + 1 + (i -1) * nc):(ns * N + i * nc)] = gu
    end

    if length(lcon3) > 0
        lcon[(1 + ns * N + N * nc):(ns * N + nc * N + num_real_bounds * N)] = lcon3
        ucon[(1 + ns * N + N * nc):(ns * N + nc * N + num_real_bounds * N)] = ucon3
    end


    c0 = zero(T)
    c  = _init_similar(s0, nvar, T)

    SparseLQDynamicModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0   = _init_similar(s0, nvar, T),
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

    nc = size(E, 1)

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s = sum(bool_vec_s)

    dense_blocks = _build_block_matrices(A, B, K, N)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0

    G  = _init_similar(Q, nc * N, nu, T)
    J  = _init_similar(Q, nc * N + num_real_bounds_s * N, nu * N, T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    _set_J1_dense!(J, G, N)

    As0 = _init_similar(s0, ns * (N + 1), T)
    LinearAlgebra.mul!(As0, block_A, s0)

    lvar = repeat(ul, N)
    uvar = repeat(uu, N)

    # Convert state variable constraints to algebraic constraints
    offset_s = N * nc
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)]
        for i in 1:N
            J[(offset_s + 1 + (i - 1) * ns):(offset_s + ns * N), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[1:(ns * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s]

            for j in 1:(N - i + 1)
                J[(offset_s + 1 + (i + j - 2) * num_real_bounds_s):(offset_s + (i + j - 1) * num_real_bounds_s), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[(1 + (j - 1) * ns):(j * ns), :][bool_vec_s, :])
            end
        end

        sl = sl[bool_vec_s]
        su = su[bool_vec_s]
    end

    lcon2 = repeat(sl, N)
    ucon2 = repeat(su, N)

    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)
    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)

    lcon = _init_similar(s0, length(dl) + length(lcon2), T)
    ucon = _init_similar(s0, length(du) + length(ucon2), T)

    lcon[1:length(dl)] = dl
    ucon[1:length(du)] = du

    if length(lcon2) > 0
        lcon[(1 + length(dl)):(length(dl) + num_real_bounds_s * N)] = lcon2
        ucon[(1 + length(du)):(length(du) + num_real_bounds_s * N)] = ucon2
    end

    nvar = nu * N
    nnzj = size(J, 1) * size(J, 2)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(J, 1)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0   = _init_similar(s0, nvar, T),
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

    nc = size(E, 1)

    dense_blocks = _build_block_matrices(A, B, K, N)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, S, Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s   = sum(bool_vec_s)

    bool_vec_u       = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds_u  = sum(bool_vec_u)


    G   = _init_similar(Q, nc * N, nu, T)
    J   = _init_similar(Q, (nc + num_real_bounds_s + num_real_bounds_u) * N, nu * N, T)
    As0 = _init_similar(s0, ns * (N + 1), T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)
    KAs0_bounds = _init_similar(s0, num_real_bounds_u * N, T)

    KBI        = _init_similar(Q, nu * N, nu, T)
    KAs0       = _init_similar(s0, nu * N, T)
    KAs0_block = _init_similar(s0, nu, T)
    KB         = _init_similar(Q, nu, nu, T)

    I_mat = _init_similar(Q, nu, nu, T)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    _set_J1_dense!(J, G, N)

    LinearAlgebra.mul!(As0, block_A, s0)

    # Convert state variable constraints to algebraic constraints
    offset_s = nc * N
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)]
        for i in 1:N
            J[(offset_s + 1 + (i - 1) * ns):(offset_s + ns * N), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[1:(ns * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] = As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s]

            for j in 1:(N - i + 1)
                J[(offset_s + 1 + (i + j - 2) * num_real_bounds_s):(offset_s + (i + j - 1) * num_real_bounds_s), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[(1 + (j - 1) * ns):(j * ns), :][bool_vec_s, :])
            end
        end

        sl = sl[bool_vec_s]
        su = su[bool_vec_s]
    end

    # Convert bounds on u to algebraic constraints
    for i in 1:N
        if i == 1
            KB = I_mat
        else
            B_row_range = (1 + (i - 2) * ns):((i - 1) * ns)
            B_sub_block = view(block_B, B_row_range, :)
            LinearAlgebra.mul!(KB, K, B_sub_block)
        end

        KBI[(1 + nu * (i - 1)):(nu * i),:] = KB
        LinearAlgebra.mul!(KAs0_block, K, As0[(1 + ns * (i - 1)):ns * i])
        KAs0[(1 + nu * (i - 1)):nu * i] = KAs0_block
    end

    offset_u = nc * N + num_real_bounds_s * N
    if num_real_bounds_u == length(ul)
        KAs0_bounds .= KAs0
        for i in 1:N
            J[(offset_u + 1 + (i - 1) * nu):(offset_u + nu * N), (1 + nu * (i - 1)):(nu * i)] = @view(KBI[1:(nu * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range              = (1 + (i - 1) * num_real_bounds_u):(i * num_real_bounds_u)
            KAs0_bounds[row_range] = KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]

            for j in 1:(N - i +1)
                J[(offset_u + 1 + (i + j - 2) * num_real_bounds_u):(offset_u + (i + j - 1) * num_real_bounds_u), (1 + nu * (i - 1)):(nu * i)] = @view(KBI[(1 + (j - 1) * nu):(j * nu), :][bool_vec_u, :])
            end
        end

        ul = ul[bool_vec_u]
        uu = uu[bool_vec_u]
    end

    lcon2 = repeat(sl, N)
    ucon2 = repeat(su, N)

    lcon3 = repeat(ul, N)
    ucon3 = repeat(uu, N)

    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)
    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)

    LinearAlgebra.axpy!(-1, KAs0_bounds, lcon3)
    LinearAlgebra.axpy!(-1, KAs0_bounds, ucon3)


    lcon = _init_similar(s0, size(J, 1), T)
    ucon = _init_similar(s0, size(J, 1), T)

    lcon[1:length(dl)] = dl
    ucon[1:length(du)] = du

    if length(lcon2) > 0
        lcon[(length(dl) + 1):(length(dl) + length(lcon2))] = lcon2
        ucon[(length(du) + 1):(length(du) + length(ucon2))] = ucon2
    end

    if length(lcon3) > 0
        lcon[(length(dl) + length(lcon2) + 1):(length(dl) + length(lcon2) + length(lcon3))] = lcon3
        ucon[(length(du) + length(ucon2) + 1):(length(du) + length(ucon2) + length(ucon3))] = ucon3
    end

    nvar = nu * N
    nnzj = size(J, 1) * size(J, 2)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(J, 1)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0   = _init_similar(s0, nvar, T),
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
        dense_blocks
    )
end

function _build_implicit_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Nothing}
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

    nc = size(E, 1)
    nvar = nu * N

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s = sum(bool_vec_s)

    G          = _init_similar(Q, nc * N, nu, T)
    Jac        = _init_similar(Q, nc * N + num_real_bounds_s * N, nu, T)
    As0        = _init_similar(s0, ns * (N + 1), T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)

    c  = _init_similar(s0, nvar, T)
    x0 = _init_similar(s0, nvar, T)

    lcon = _init_similar(s0, nc * N + num_real_bounds_s * N, T)
    ucon = _init_similar(s0, nc * N + num_real_bounds_s * N, T)

    SJ1  = _init_similar(s0, nc, nu, T)
    SJ2  = _init_similar(s0, num_real_bounds_s, nu, T)
    SJ3  = _init_similar(s0, 0, nu, T)

    dense_blocks = _build_block_matrices(A, B, K, N)
    block_A      = dense_blocks.A
    block_B      = dense_blocks.B

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0
    c .= H_blocks.c

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    Jac[1:nc * N, :] = G

    LinearAlgebra.mul!(As0, block_A, s0)

    lvar = repeat(ul, N)
    uvar = repeat(uu, N)

    # Convert state variable constraints to algebraic constraints
    offset_s = N * nc
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)]
        Jac[(1 + N * nc):(N * nc + ns * N), :] = block_B
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s]
            Jac[(1 + offset_s + (i - 1) * num_real_bounds_s):offset_s + i * num_real_bounds_s] = @view block_B[(1 + (i - 1) * ns):(i*ns), :][bool_vec_s, :]
        end
        sl = sl[bool_vec_s]
        su = su[bool_vec_s]
    end

    lcon2 = repeat(sl, N)
    ucon2 = repeat(su, N)

    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)
    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)

    lcon[1:length(dl)] = dl
    ucon[1:length(du)] = du

    if length(lcon2) > 0
        lcon[(1 + length(dl)):(length(dl) + num_real_bounds_s * N)] = lcon2
        ucon[(1 + length(du)):(length(du) + num_real_bounds_s * N)] = ucon2
    end

    nnzj = size(Jac, 1) * size(H, 2)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(Jac, 1)

    J = LQJacobianOperator{T, V, M}(
        Jac, N, nu, nc, num_real_bounds_s, 0,
        SJ1, SJ2, SJ3
    )

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0   = x0,
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
        dense_blocks
    )
end

function _build_implicit_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}
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

    nc = size(E, 1)

    dense_blocks = _build_block_matrices(A, B, K, N)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, S, Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s   = sum(bool_vec_s)

    bool_vec_u       = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds_u  = sum(bool_vec_u)


    G   = _init_similar(Q, nc * N, nu, T)
    Jac = _init_similar(Q, (nc + num_real_bounds_s + num_real_bounds_u) * N, nu, T)

    As0 = _init_similar(s0, ns * (N + 1), T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)
    KAs0_bounds = _init_similar(s0, num_real_bounds_u * N, T)

    KBI        = _init_similar(Q, nu * N, nu, T)
    KAs0       = _init_similar(s0, nu * N, T)
    KAs0_block = _init_similar(s0, nu, T)
    KB         = _init_similar(Q, nu, nu, T)

    lcon = _init_similar(s0, size(Jac, 1), T)
    ucon = _init_similar(s0, size(Jac, 1), T)

    I_mat = _init_similar(Q, nu, nu, T)

    SJ1   = _init_similar(s0, nc, nu, T)
    SJ2   = _init_similar(s0, num_real_bounds_s, nu, T)
    SJ3   = _init_similar(s0, num_real_bounds_u, nu, T)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    Jac[1:nc * N, :] = G

    LinearAlgebra.mul!(As0, block_A, s0)

    # Convert state variable constraints to algebraic constraints
    offset_s = nc * N
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)]
        Jac[(1 + N * nc):(N * nc + ns * N), :] = block_B
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s]
            Jac[(1 + offset_s + (i - 1) * num_real_bounds_s):offset_s + i * num_real_bounds_s] = @view block_B[(1 + (i - 1) * ns):(i*ns), :][bool_vec_s, :]
        end
        sl = sl[bool_vec_s]
        su = su[bool_vec_s]
    end

    # Convert bounds on u to algebraic constraints
    for i in 1:N
        if i == 1
            KB = I_mat
        else
            B_row_range = (1 + (i - 2) * ns):((i - 1) * ns)
            B_sub_block = view(block_B, B_row_range, :)
            LinearAlgebra.mul!(KB, K, B_sub_block)
        end

        KBI[(1 + nu * (i - 1)):(nu * i),:] = KB
        LinearAlgebra.mul!(KAs0_block, K, As0[(1 + ns * (i - 1)):ns * i])
        KAs0[(1 + nu * (i - 1)):nu * i] = KAs0_block
    end

    offset_u = nc * N + num_real_bounds_s * N
    if num_real_bounds_u == length(ul)
        KAs0_bounds .= KAs0
        Jac[(1 + offset_u):(offset_u + nu * N), :] = KBI
    else
        for i in 1:N
            row_range              = (1 + (i - 1) * num_real_bounds_u):(i * num_real_bounds_u)
            KAs0_bounds[row_range] = KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]

            Jac[(1 + offset_u + (i - 1) * num_real_bounds_u):(offset_u + i * num_real_bounds_u), :] = @view KBI[(1 + (i - 1) * nu):(i * nu), :][bool_vec_u, :]
        end

        ul = ul[bool_vec_u]
        uu = uu[bool_vec_u]
    end

    lcon2 = repeat(sl, N)
    ucon2 = repeat(su, N)

    lcon3 = repeat(ul, N)
    ucon3 = repeat(uu, N)

    LinearAlgebra.axpy!(-1, As0_bounds, lcon2)
    LinearAlgebra.axpy!(-1, As0_bounds, ucon2)

    LinearAlgebra.axpy!(-1, KAs0_bounds, lcon3)
    LinearAlgebra.axpy!(-1, KAs0_bounds, ucon3)

    lcon[1:length(dl)] = dl
    ucon[1:length(du)] = du

    if length(lcon2) > 0
        lcon[(length(dl) + 1):(length(dl) + length(lcon2))] = lcon2
        ucon[(length(du) + 1):(length(du) + length(ucon2))] = ucon2
    end

    if length(lcon3) > 0
        lcon[(length(dl) + length(lcon2) + 1):(length(dl) + length(lcon2) + length(lcon3))] = lcon3
        ucon[(length(du) + length(ucon2) + 1):(length(du) + length(ucon2) + length(ucon3))] = ucon3
    end

    nvar = nu * N
    nnzj = size(Jac, 1) * size(H, 1)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(Jac, 1)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    J = LQJacobianOperator{T, V, M}(
        Jac, N, nu, nc, num_real_bounds_s, num_real_bounds_u,
        SJ1, SJ2, SJ3
    )

    DenseLQDynamicModel(
        NLPModels.NLPModelMeta(
        nvar,
        x0   = _init_similar(s0, nvar, T),
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
        dense_blocks
    )
end

function _build_block_matrices(
    A::M, B::M, K, N
) where {T, M <: AbstractMatrix{T}}

    ns = size(A, 2)
    nu = size(B, 2)

    if K == nothing
        K = _init_similar(A, nu, ns, T)
    end

    # Define block matrices
    block_A = _init_similar(A, ns * (N + 1), ns, T)
    block_B = _init_similar(B, ns * N, nu, T)

    A_k = copy(A)
    BK  = _init_similar(A, ns, ns, T)


    AB_klast = _init_similar(A, size(B, 1), size(B, 2), T)
    AB_k     = _init_similar(A, size(B, 1), size(B, 2), T)


    block_B[1:ns, :] = B

    for i in 1:ns
        block_A[i, i] = T(1)
    end

    LinearAlgebra.mul!(BK, B, K)
    LinearAlgebra.axpy!(1, BK, A_k)

    A_klast  = copy(A_k)
    A_knext  = copy(A_k)

    block_A[(ns + 1):ns *2, :] = A_k
    LinearAlgebra.mul!(AB_k, A_k, B, 1, 0)

    block_B[(1 + ns):2 * ns, :] = AB_k
    AB_klast = copy(AB_k)

    # Fill the A and B matrices
    for i in 2:(N - 1)

        LinearAlgebra.mul!(AB_k, A_k, AB_klast)

        LinearAlgebra.mul!(A_knext, A_k, A_klast)

        block_A[(ns * i + 1):ns * (i + 1),:] = A_knext

        block_B[(1 + (i) * ns):((i + 1) * ns), :] = AB_k

        AB_klast = copy(AB_k)
        A_klast  = copy(A_knext)
    end

    LinearAlgebra.mul!(A_knext, A_k, A_klast)

    block_A[(ns * N + 1):ns * (N + 1), :] = A_knext

    DenseLQDynamicBlocks{T, M}(
        block_A,
        block_B
    )
end


function _build_H_blocks(Q, R, block_A::M, block_B::M, S, Qf, K, s0, N) where {T, M <: AbstractMatrix{T}}

    ns = size(Q, 1)
    nu = size(R, 1)

    if K == nothing
        K = _init_similar(Q, nu, ns, T)
    end

    H = _init_similar(block_A, nu * N, nu * N, T)

    # quad term refers to the summation of Q, K^T RK, SK, and K^T S^T that is left and right multiplied by B in the Hessian
    quad_term    = _init_similar(Q, ns, ns, T)

    quad_term_B  = _init_similar(block_B, size(block_B, 1), size(block_B, 2), T)
    QfB          = _init_similar(block_B, size(block_B, 1), size(block_B, 2), T)

    quad_term_AB = _init_similar(block_A, ns, nu, T)
    QfAB         = _init_similar(block_A, ns, nu, T)

    RK_STB       = _init_similar(block_B, nu, nu, T)
    BQB          = _init_similar(block_B, nu, nu, T)
    BQfB         = _init_similar(block_B, nu, nu, T)
    SK           = _init_similar(Q, ns, ns, T)
    RK           = _init_similar(Q, nu, ns, T)
    KTRK         = _init_similar(Q, ns, ns, T)
    RK_ST        = _init_similar(Q, nu, ns, T)
    As0          = _init_similar(s0, ns * (N + 1), T)

    QB_block_vec = _init_similar(quad_term_B, ns * (N + 1), nu, T)
    h            = _init_similar(s0, nu * N, T)
    h0           = zero(T)

    As0QB        = _init_similar(s0, nu, T)
    QAs0         = _init_similar(s0, ns, T)
    As0S         = _init_similar(s0, nu, T)
    KTRKAs0      = _init_similar(s0, ns, T)
    SKAs0        = _init_similar(s0, ns, T)

    LinearAlgebra.mul!(SK, S, K)
    LinearAlgebra.mul!(RK, R, K)
    LinearAlgebra.mul!(KTRK, K', RK)

    LinearAlgebra.axpy!(1.0, Q, quad_term)
    LinearAlgebra.axpy!(1.0, SK, quad_term)
    LinearAlgebra.axpy!(1.0, SK', quad_term)
    LinearAlgebra.axpy!(1.0, KTRK, quad_term)

    LinearAlgebra.copyto!(RK_ST, RK)
    LinearAlgebra.axpy!(1.0, S', RK_ST)

    LinearAlgebra.mul!(As0, block_A, s0)

    for i in 1:N
        B_row_range = (1 + (i - 1) * ns):(i * ns)
        B_sub_block = view(block_B, B_row_range, :)

        LinearAlgebra.mul!(quad_term_AB, quad_term, B_sub_block)
        LinearAlgebra.mul!(QfAB, Qf, B_sub_block)

        quad_term_B[(1 + (i - 1) * ns):(i * ns), :]  = quad_term_AB
        QfB[(1 + (i - 1) * ns):(i * ns), :] = QfAB

        for j in 1:(N + 1 - i)
            right_block = block_B[(1 + (j - 1 + i - 1) * ns):((j + i - 1)* ns), :]
            LinearAlgebra.mul!(BQB, quad_term_AB', right_block)
            LinearAlgebra.mul!(BQfB, QfAB', right_block)


            for k in 1:(N - j - i + 2)
                row_range = (1 + nu * (k + (j - 1) - 1)):(nu * (k + (j - 1)))
                col_range = (1 + nu * (k - 1)):(nu * k)

                if k == N - j - i + 2
                    view(H, row_range, col_range) .+= BQfB
                else
                    view(H, row_range, col_range) .+= BQB
                end
            end

        end
        LinearAlgebra.mul!(RK_STB, RK_ST, B_sub_block)
        for m in 1:(N - i)
            row_range = (1 + nu * (m - 1 + i)):(nu * (m + i))
            col_range = (1 + nu * (m - 1)):(nu * m)

            view(H, row_range, col_range) .+= RK_STB
        end

        view(H, (1 + nu * (i - 1)):nu * i, (1 + nu * (i - 1)):nu * i) .+= R
    end


    for i in 1:N
        fill!(QB_block_vec, T(0))
        rows_QB           = 1:(ns * (N - i))
        rows_QfB          = (1 + ns * (N - i)):(ns * (N - i + 1))

        QB_block_vec[(1 + ns * i):(ns * N), :]     = quad_term_B[rows_QB, :]
        QB_block_vec[(1 + ns * N):ns * (N + 1), :] = QfB[rows_QfB, :]

        LinearAlgebra.mul!(As0QB, QB_block_vec', As0)
        LinearAlgebra.mul!(As0S, RK_ST, As0[(ns * (i - 1) + 1):ns * i])

        h[(1 + nu * (i - 1)):nu * i] = As0QB
        view(h, (1 + nu * (i - 1)):nu * i) .+= As0S

        LinearAlgebra.mul!(QAs0, Q, As0[(ns * (i - 1) + 1):ns * i])
        LinearAlgebra.mul!(KTRKAs0, KTRK, As0[(ns * (i - 1) + 1):ns * i])
        LinearAlgebra.mul!(SKAs0, SK, As0[(ns * (i - 1) + 1):ns * i])

        h0 += LinearAlgebra.dot(As0[(ns * (i - 1) + 1):ns * i], QAs0)
        h0 += LinearAlgebra.dot(As0[(ns * (i - 1) + 1):ns * i], KTRKAs0)
        h0 += T(2) * LinearAlgebra.dot(As0[(ns * (i - 1) + 1):ns * i], SKAs0)
    end

    LinearAlgebra.mul!(QAs0, Qf, As0[(ns * N + 1):ns * (N + 1)])
    LinearAlgebra.mul!(KTRKAs0, KTRK, As0[(ns * N + 1):ns * (N + 1)])
    LinearAlgebra.mul!(SKAs0, SK, As0[(ns * N + 1):ns * (N + 1)])

    h0 += LinearAlgebra.dot(As0[(ns * N + 1):ns * (N + 1)], QAs0)

    return (H = H, c = h, c0 = h0 / T(2))
end


function _set_G_blocks!(G, dl, du, block_B::M, block_A::M, s0, E, F, K::MK, N) where {T, M <: AbstractMatrix{T}, MK <: Nothing}
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    G[1:nc, :] = F

    EB   = _init_similar(block_B, nc, nu, T)
    As0  = _init_similar(s0, ns * (N + 1), T)
    EAs0 = _init_similar(s0, nc, T)

    LinearAlgebra.mul!(As0, block_A, s0)

    for i in 1:N
        if i != N
            B_row_range = (1 + (i - 1) * ns):(i * ns)
            B_sub_block = view(block_B, B_row_range, :)

            LinearAlgebra.mul!(EB, E, B_sub_block)
            G[(1 + nc * i):(nc * (i + 1)), :] = EB
        end

        LinearAlgebra.mul!(EAs0, E, As0[(ns * (i - 1) + 1):ns * i])

        dl[(1 + nc * (i - 1)):nc * i] .-= EAs0
        du[(1 + nc * (i - 1)):nc * i] .-= EAs0
    end

end

function _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K::MK, N) where {T, MK <: AbstractMatrix{T}}
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    G[1:nc, :] = F

    E_FK = _init_similar(E, nc, ns, T)
    FK   = _init_similar(E, nc, ns, T)
    EB   = _init_similar(E, nc, nu, T)
    As0  = _init_similar(s0, ns * (N + 1), T)
    EAs0 = _init_similar(s0, nc, T)

    LinearAlgebra.copyto!(E_FK, E)
    LinearAlgebra.mul!(FK, F, K)
    LinearAlgebra.axpy!(1.0, FK, E_FK)

    LinearAlgebra.mul!(As0, block_A, s0)

    for i in 1:N
        if i != N
            B_row_range = (1 + (i - 1) * ns):(i * ns)
            B_sub_block = view(block_B, B_row_range, :)

            LinearAlgebra.mul!(EB, E_FK, B_sub_block)
            G[(1 + nc * i):(nc * (i + 1)), :] = EB
        end


        LinearAlgebra.mul!(EAs0, E_FK, As0[(ns * (i - 1) + 1):ns * i])

        dl[(1 + nc * (i - 1)):nc * i] .-= EAs0
        du[(1 + nc * (i - 1)):nc * i] .-= EAs0
    end

end

function _set_J1_dense!(J1, G, N)
    nu = size(G, 2)
    nc = Int(size(G, 1) / N)

    for i in 1:N
        col_range = (1 + nu * (i - 1)):(nu * i)
        J1[(1 + nc * (i - 1)):nc * N, col_range] = G[1:((N - i + 1) * nc),:]
    end

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

    N    = dnlp.N
    ns   = dnlp.ns
    nu   = dnlp.nu
    K    = dnlp.K

    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B

    v = solver_status.solution

    As0 = zeros(T, ns * (N + 1))
    Bv  = zeros(T, ns)
    s   = zeros(T, ns * (N + 1))

    for i in 1:N
        B_row_range = (1 + (i - 1) * ns):(i * ns)
        B_sub_block = view(block_B, B_row_range, :)

        for j in 1:(N - i + 1)
            v_sub_vec   = v[(1 + nu * (j - 1)):nu * j]
            LinearAlgebra.mul!(Bv, B_sub_block, v_sub_vec)

            s[(1 + ns * (i + j - 1)):(ns * (i + j))] .+= Bv
        end
    end

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.axpy!(1, As0, s)

    Ks = _init_similar(dnlp.s0, ns, T)
    u = copy(v)
    for i in 1:N
        LinearAlgebra.mul!(Ks, K, s[(1 + ns * (i - 1)):ns * i])
        u[(1 + nu * (i - 1)):nu * i] .+= Ks
    end

    return u
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

    N    = dnlp.N
    ns   = dnlp.ns
    nu   = dnlp.nu

    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B

    v = solver_status.solution

    As0 = zeros(T, ns * (N + 1))
    Bv  = zeros(T, ns)
    s   = zeros(T, ns * (N + 1))

    for i in 1:N
        B_row_range = (1 + (i - 1) * ns):(i * ns)
        B_sub_block = view(block_B, B_row_range, :)

        for j in 1:(N - i + 1)
            v_sub_vec   = v[(1 + nu * (j - 1)):nu * j]
            LinearAlgebra.mul!(Bv, B_sub_block, v_sub_vec)

            s[(1 + ns * (i + j - 1)):(ns * (i + j))] .+= Bv
        end
    end

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.axpy!(1, As0, s)

    return s
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
    _set_sparse_H!(H_colptr, H_rowval, H_nzval, Q, R, N; Qf = Q, S = zeros(T, size(Q, 1), size(R, 1))

set the data needed to build a SparseArrays.SparseMatrixCSC matrix. H_colptr, H_rowval, and H_nzval
are set so that they can be passed to SparseMatrixCSC() to obtain the `H` matrix such that
 z^T H z = sum_{i=1}^{N-1} s_i^T Q s + sum_{i=1}^{N-1} u^T R u + s_N^T Qf s_n .
"""
function _set_sparse_H!(
    H_colptr, H_rowval, H_nzval,
    Q::M, R::M, N;
    Qf::M = Q,
    S::M = zeros(T, size(Q, 1), size(R, 1))
) where {T, M <: AbstractMatrix{T}}

    ns = size(Q, 1)
    nu = size(R, 1)


    for i in 1:N
        for j in 1:ns
            H_nzval[(1 + (i - 1) * (ns^2 + nu * ns) + (j - 1) * (ns + nu)):(ns * j + nu * (j - 1) + (i - 1) * (ns^2 + nu * ns))]  = @view Q[:, j]
            H_nzval[(1 + (i - 1) * (ns^2 + nu * ns) + j * ns + (j - 1) * nu):((i - 1) * (ns^2 + nu * ns) + j * (ns + nu))] = @view S[j, :]
            H_rowval[(1 + (i - 1) * (ns^2 + nu * ns) + (j - 1) * ns + (j - 1) * nu):(ns * j + nu * (j - 1) + (i - 1) * (ns^2 + nu * ns))] = (1 + (i - 1) * ns):ns * i
            H_rowval[(1 + (i - 1) * (ns^2 + nu * ns) + j * ns + (j - 1) * nu ):((i - 1) * (ns^2 + nu * ns) + j * (ns + nu))] =(1 + (N + 1) * ns + nu * (i - 1)):((N + 1) * ns + nu * i)
            H_colptr[((i - 1) * ns + j)] = 1 + (ns + nu) * (j - 1) + (i - 1) * (ns * nu + ns * ns)
        end
    end

    for j in 1:ns
        H_nzval[(1 + N * (ns^2 + nu * ns) + (j - 1) * ns):(ns * j + N * (ns^2 + nu * ns))]  = @view Qf[:, j]
        H_rowval[(1 + N * (ns^2 + nu * ns) + (j - 1) * ns):(ns * j + N * (ns^2 + nu * ns))] = (1 + N * ns):((N + 1) * ns)
        H_colptr[(N * ns + j)] = 1 + ns * (j - 1) + N * (ns * nu + ns * ns)
    end

    offset = ns^2 * (N + 1) + ns * nu * N
    for i in 1:N
        for j in 1:nu
            H_nzval[(1 + offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * (nu + ns)):(offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * nu + j * ns)]  = @view S[:,j]
            H_nzval[(1 + offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * nu + j * ns):(offset + (i - 1) * (nu^2 + ns * nu) +  j * (ns + nu ))]      = @view R[:, j]
            H_rowval[(1 + offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * (nu + ns)):(offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * nu + j * ns)] = (1 + (i - 1) * ns):i * ns
            H_rowval[(1 + offset + (i - 1) * (nu^2 + ns * nu) + (j - 1) * nu + j * ns):(offset + (i - 1) * (nu^2 + ns * nu) +  j * (ns + nu ))]     = (1 + (N + 1) * ns + (i - 1) * nu):((N + 1) * ns + i * nu)
            H_colptr[(N + 1) * ns + (i - 1) * nu + j] = 1 + offset + (ns + nu) * (j - 1) + (nu^2 + ns * nu) * (i - 1)
        end
    end

    H_colptr[ns * (N + 1) + nu * N + 1] = length(H_nzval) + 1
end

"""
    _set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, bool_vec, N, nb)
    _set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, N)

set the data needed to build a SparseArrays.SparseMatrixCSC matrix. J_colptr, J_rowval, and J_nzval
are set so that they can be passed to SparseMatrixCSC() to obtain the Jacobian, `J`. The Jacobian
contains the data for the following constraints:

As_i + Bu_i = s_{i + 1}
gl <= Es_i + Fu_i <= get_u

If `K` is defined, then this matrix also contains the constraints
ul <= Kx_i + v_i <= uu
"""
function _set_sparse_J!(
    J_colptr, J_rowval, J_nzval,
    A, B, E, F, K::MK, bool_vec,
    N, nb
) where {T, MK <: AbstractMatrix{T}}
    # nb = num_real_bounds

    ns = size(A, 2)
    nu = size(B, 2)
    nc = size(E, 1)

    I_mat = _init_similar(A, nu, nu)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    # Set the first block column of A, E, and K
    for j in 1:ns
        J_nzval[(1 + (j - 1) * (ns + nc + nb)):((j - 1) * (nc + nb) + j * ns)]      = @view A[:, j]
        J_nzval[(1 + (j - 1) * (nc + nb) + j * ns):(j * (ns + nc) + (j - 1) * nb)]  = @view E[:, j]
        J_nzval[(1 + j * (ns + nc) + (j - 1) * nb):(j * (ns + nc + nb))]            = @view K[:, j][bool_vec]
        J_rowval[(1 + (j - 1) * (ns + nc + nb)):((j - 1) * (nc + nb) + j * ns)]     = 1:ns
        J_rowval[(1 + (j - 1) * (nc + nb) + j * ns):(j * (ns + nc) + (j - 1) * nb)] = (1 + ns * N):(nc + ns * N)
        J_rowval[(1 + j * (ns + nc) + (j - 1) * nb):(j * (ns + nc + nb))]           = (1 + (ns + nc) * N):((ns + nc) * N + nb)
        J_colptr[j] = 1 + (j - 1) * (ns + nc + nb)
    end

    # Set the remaining block columns corresponding to states: -I, A, E, K
    for i in 2:N
        offset = (i - 1) * ns * (ns + nc + nb) + (i - 2) * ns
        for j in 1:ns
            J_nzval[1 + offset + (j - 1) * (ns + nc + nb + 1)]  = T(-1)
            J_nzval[(1 + offset + (j - 1) * (ns + nc + nb) + j):(offset + j * ns + (j - 1) * (nc + nb) + j)]      = @view A[:, j]
            J_nzval[(1 + offset + j * ns + (j - 1) * (nc + nb) + j):(offset + j * (ns + nc) + (j - 1) * nb + j)]  = @view E[:, j]
            J_nzval[(1 + offset + j * (ns + nc) + (j - 1) * nb + j):(offset + j * (ns + nc + nb) + j)]            = @view K[:, j][bool_vec]
            J_rowval[1 + offset + (j - 1) * (ns + nc + nb + 1)] = ns * (i - 2) + j
            J_rowval[(1 + offset + (j - 1) * (ns + nc + nb) + j):(offset + j * ns + (j - 1) * (nc + nb) + j)]     = (1 + (i - 1) * ns):(i * ns)
            J_rowval[(1 + offset + j * ns + (j - 1) * (nc + nb) + j):(offset + j * (ns + nc) + (j - 1) * nb + j)] = (1 + N * ns + (i - 1) * nc):(N * ns + i * nc)
            J_rowval[(1 + offset + j * (ns + nc) + (j - 1) * nb + j):(offset + j * (ns + nc + nb) + j)]           = (1 + N * (ns + nc) + (i - 1) * nb):(N * (ns + nc) + i * nb)
            J_colptr[(i - 1) * ns + j] = 1 + (j - 1) * (ns + nc + nb + 1) + offset
        end
    end

    # Set the column corresponding to states at N + 1, which are a single block of -I
    for j in 1:ns
        J_nzval[j + ns * (ns + nc + nb + 1) * N - ns]  = T(-1)
        J_rowval[j + ns * (ns + nc + nb + 1) * N - ns] = j + (N - 1) * ns
        J_colptr[ns * N + j] = 1 + ns * (ns + nc + nb + 1) * N - ns + (j - 1)
    end

    # Set the remaining block columns corresponding to inputs: B, F, I
    nscol_offset = N * (ns^2 + nc * ns + nb * ns + ns)
    for i in 1:N
        offset = (i - 1) * (nu * ns + nu * nc + nb) + nscol_offset
        bool_offset = 0
        for j in 1:nu
            J_nzval[(1 + offset + (j - 1) * (ns + nc) + bool_offset):(offset + j * ns + (j - 1) * nc + bool_offset)]  = @view B[:, j]
            J_nzval[(1 + offset + j * ns + (j - 1) * nc + bool_offset):(offset + j * (ns + nc) + bool_offset)]  = @view F[:, j]
            if bool_vec[j]
                J_nzval[1 + offset + j * (ns + nc) + bool_offset]  = T(1)
                J_rowval[1 + offset + j * (ns + nc) + bool_offset] = (N * (ns + nc) + (i - 1) * nb + 1 + (bool_offset))
            end
            J_rowval[(1 + offset + (j - 1) * (ns + nc) + bool_offset):(offset + j * ns + (j - 1) * nc + bool_offset)] = (1 + (i - 1) * ns):i * ns
            J_rowval[(1 + offset + j * ns + (j - 1) * nc + bool_offset):(offset + j * (ns + nc) + bool_offset)] = (1 + N * ns + (i - 1) * nc):(N * ns + i * nc)
            J_colptr[(ns * (N + 1) + (i - 1) * nu + j)] = 1 + offset + (j - 1) * (ns + nc) + bool_offset

            bool_offset += bool_vec[j]
        end
    end

    J_colptr[ns * (N + 1) + nu * N + 1] = length(J_nzval) + 1
end


function _set_sparse_J!(
    J_colptr, J_rowval, J_nzval,
    A::M, B::M, E, F, K::MK, N
) where {T, M <: AbstractMatrix{T}, MK <: Nothing}
    # nb = num_real_bounds

    ns = size(A, 2)
    nu = size(B, 2)
    nc = size(E, 1)


    # Set the first block column of A, E, and K
    for j in 1:ns
        J_nzval[(1 + (j - 1) * (ns + nc)):((j - 1) * nc + j * ns)]  = @view A[:, j]
        J_nzval[(1 + (j - 1) * nc + j * ns):(j * (ns + nc))]        = @view E[:, j]
        J_rowval[(1 + (j - 1) * (ns + nc)):((j - 1) * nc + j * ns)] = 1:ns
        J_rowval[(1 + (j - 1) * nc + j * ns):(j * (ns + nc))]       = (1 + ns * N):(nc + ns * N)
        J_colptr[j] = 1 + (j - 1) * (ns + nc)
    end

    # Set the remaining block columns corresponding to states: -I, A, E, K
    for i in 2:N
        offset = (i - 1) * ns * (ns + nc) + (i - 2) * ns
        for j in 1:ns
            J_nzval[1 + offset + (j - 1) * (ns + nc + 1)]  = T(-1)
            J_nzval[(1 + offset + (j - 1) * (ns + nc) + j):(offset + j * ns + (j - 1) * nc + j)]  = @view A[:, j]
            J_nzval[(1 + offset + j * ns + (j - 1) * nc + j):(offset + j * (ns + nc) + j)]        = @view E[:, j]
            J_rowval[1 + offset + (j - 1) * (ns + nc + 1)] = ns * (i - 2) + j
            J_rowval[(1 + offset + (j - 1) * (ns + nc) + j):(offset + j * ns + (j - 1) * nc + j)] = (1 + (i - 1) * ns):(i * ns)
            J_rowval[(1 + offset + j * ns + (j - 1) * nc + j):(offset + j * (ns + nc) + j)]       = (1 + N * ns + (i - 1) * nc):(N * ns + i * nc)
            J_colptr[(i - 1) * ns + j] = 1 + (j - 1) * (ns + nc + 1) + offset
        end
    end

    # Set the column corresponding to states at N + 1, which are a single block of -I
    for j in 1:ns
        J_nzval[j + ns * (ns + nc + 1) * N - ns]  = T(-1)
        J_rowval[j + ns * (ns + nc + 1) * N - ns] = j + (N - 1) * ns
        J_colptr[ns * N + j] = 1 + ns * (ns + nc + 1) * N - ns + (j - 1)
    end

    # Set the remaining block columns corresponding to inputs: B, F, I
    nscol_offset = N * (ns^2 + nc * ns + ns)
    for i in 1:N
        offset = (i - 1) * (nu * ns + nu * nc) + nscol_offset
        for j in 1:nu
            J_nzval[(1 + offset + (j - 1) * (ns + nc)):(offset + j * ns + (j - 1) * nc)]  = @view B[:, j]
            J_nzval[(1 + offset + j * ns + (j - 1) * nc):(offset + j * (ns + nc))]        = @view F[:, j]
            J_rowval[(1 + offset + (j - 1) * (ns + nc)):(offset + j * ns + (j - 1) * nc)] = (1 + (i - 1) * ns):i * ns
            J_rowval[(1 + offset + j * ns + (j - 1) * nc):(offset + j * (ns + nc))]       = (1 + N * ns + (i - 1) * nc):(N * ns + i * nc)
            J_colptr[(ns * (N + 1) + (i - 1) * nu + j)] = 1 + offset + (j - 1) * (ns + nc)
        end
    end

    J_colptr[ns * (N + 1) + nu * N + 1] = length(J_nzval) + 1
end

function _cmp_arr(op, A, B)
    for i = 1:length(A)
        !op(A[i], B[i]) && return true
    end
    return false
end

function _init_similar(mat, dim1::Number, dim2::Number, T=eltype(mat))
    new_mat = similar(mat, dim1, dim2); fill!(new_mat, zero(T))
    return new_mat
end

function _init_similar(mat, dim1::Number, T=eltype(mat))
    new_mat = similar(mat, dim1); fill!(new_mat, zero(T))
    return new_mat
end


function LinearAlgebra.mul!(y::V,
    Jac::LQJacobianOperator{T, V, M},
    x::V
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    fill!(y, zero(T))

    J   = Jac.truncated_jac
    N   = Jac.N
    nu  = Jac.nu
    nc  = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    for i in 1:N
        sub_B1 = @view J[(1 + (i - 1) * nc):(i * nc), :]
        sub_B2 = @view J[(1 + nc * N + (i - 1) * nsc):(nc * N + i * nsc), :]
        sub_B3 = @view J[(1 + nc * N + nsc * N + (i - 1) * nuc):(nc * N + nsc * N + nuc * i), :]

        for j in 1:(N - i + 1)
            sub_x = view(x, (1 + (j - 1) * nu):(j * nu))
            LinearAlgebra.mul!(view(y, (1 + nc * (j + i - 2)):(nc * (j + i - 1) )), sub_B1, sub_x, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nc * N + nsc * (j + i - 2)):(nc * N + nsc * (j + i - 1))), sub_B2, sub_x, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nc * N + nsc * N + nuc * (j + i- 2)):(nc * N + nsc * N + nuc * (j + i - 1))), sub_B3, sub_x, 1, 1)
        end
    end
end

function LinearAlgebra.mul!(
    y::V,
    Jac::LinearOperators.AdjointLinearOperator{T, LQJacobianOperator{T, V, M}},
    x::V
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    fill!(y, zero(T))

    J   = get_jacobian(Jac).truncated_jac
    N   = get_jacobian(Jac).N
    nu  = get_jacobian(Jac).nu
    nc  = get_jacobian(Jac).nc
    nsc = get_jacobian(Jac).nsc
    nuc = get_jacobian(Jac).nuc

    for i in 1:N
        sub_B1 = @view J[(1 + (i - 1) * nc):(i * nc), :]
        sub_B2 = @view J[(1 + nc * N + (i - 1) * nsc):(nc * N + i * nsc), :]
        sub_B3 = @view J[(1 + nc * N + nsc * N + (i - 1) * nuc):(nc * N + nsc * N + nuc * i), :]

        for j in 1:(N - i + 1)
            x1 = view(x, (1 + (j + i - 2) * nc):((j + i - 1) * nc))
            x2 = view(x, (1 + nc * N + (j + i - 2) * nsc):(nc * N + (j + i - 1) * nsc))
            x3 = view(x, (1 + nc * N + nsc * N + (j + i - 2) * nuc):(nc * N + nsc * N + (j + i - 1) * nuc))

            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j )), sub_B1', x1, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j )), sub_B2', x2, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j )), sub_B3', x3, 1, 1)
        end
    end
end


"""
    get_jacobian(lqdm::DenseLQDynamicModel) -> LQJacobianOperator
    get_jacobian(Jac::AdjointLinearOpeartor{T, LQJacobianOperator}) -> LQJacobianOperator

Gets the `LQJacobianOperator` from `DenseLQDynamicModel` (if the `QPdata` contains a `LQJacobian Operator`)
or returns the `LQJacobian Operator` from the adjoint of the `LQJacobianOperator`
"""
function get_jacobian(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK}
) where {T, V, M1, M2, M3, M4, MK}
    return lqdm.data.A
end

function get_jacobian(
    Jac::LinearOperators.AdjointLinearOperator{T, LQJacobianOperator{T, V, M}}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    return Jac'
end

function Base.length(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    return length(Jac.truncated_jac)
end

function Base.size(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    return size(Jac.truncated_jac)
end

function Base.eltype(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    return T
end

function Base.isreal(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    return isreal(Jac.truncated_jac)
end

function Base.show(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    show(Jac.truncated_jac)
end

function Base.display(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    display(Jac.truncated_jac)
end
"""
    LinearOperators.reset!(Jac::LQJacobianOperator{T, V, M})

Resets the values of attributes `SJ1`, `SJ2`, and `SJ3` to zero
"""
function LinearOperators.reset!(
    Jac::LQJacobianOperator{T, V, M}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    fill!(Jac.SJ1, T(0))
    fill!(Jac.SJ2, T(0))
    fill!(Jac.SJ3, T(0))
end

function NLPModels.jac_op(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK}, x::V
) where {T, V <: AbstractVector{T}, M1, M2 <: LQJacobianOperator, M3, M4, MK}
    return lqdm.data.A
end

"""
    add_jtsj!(H::M, Jac::LQJacobianOperator{T, V, M}, Σ::V, alpha::Number = 1, beta::Number = 1)

Generates `Jac' Σ Jac` and adds it to the matrix `H`.

`alpha` and `beta` are scalar multipliers such `beta H + alpha Jac' Σ Jac` is stored in `H`, overwriting the existing value of `H`
"""
function add_jtsj!(
    H::M,
    Jac::LQJacobianOperator{T, V, M},
    Σ::V,
    alpha::Number = 1,
    beta::Number = 1
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}

    J   = Jac.truncated_jac
    N   = Jac.N
    nu  = Jac.nu
    nc  = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    ΣJ1 = Jac.SJ1
    ΣJ2 = Jac.SJ2
    ΣJ3 = Jac.SJ3

    LinearAlgebra.lmul!(beta, H)

    for i in 1:N
        J1_left_range = (1 + (i - 1) * nc):(i * nc)
        J2_left_range = (1 + nc * N + (i - 1) * nsc):(nc * N + i * nsc)
        J3_left_range = (1 + (nc + nsc) * N + (i - 1) * nuc):((nc + nsc) * N + i * nuc)
        left_block1 = view(J, J1_left_range, :)
        left_block2 = view(J, J2_left_range, :)
        left_block3 = view(J, J3_left_range, :)

        for j in 1:(N + 1 - i)
            J1_right_range = (1 + (j + i - 2) * nc):((j + i - 1) * nc)
            J2_right_range = (1 + nc * N + (j + i - 2) * nsc):(nc * N + (j + i - 1) * nsc)
            J3_right_range = (1 + (nc + nsc) * N + (j + i - 2) * nuc):((nc + nsc) * N + (j + i - 1) * nuc)

            right_block1 = view(J, J1_right_range, :)
            right_block2 = view(J, J2_right_range, :)
            right_block3 = view(J, J3_right_range, :)

            for k in 1:(N - j - i + 2)
                Σ_range1 = (1 + (k + i + j + - 3) * nc):((k + i + j - 2) * nc)
                Σ_range2 = (1 + nc * N + (k + i + j - 3) * nsc):(nc * N + (k + i + j - 2) * nsc)
                Σ_range3 = (1 + (nc + nsc) * N + (k + i + j - 3) * nuc):((nc + nsc) * N + (k + i + j - 2) * nuc)
                ΣJ1 .= right_block1 .* view(Σ, Σ_range1)
                ΣJ2 .= right_block2 .* view(Σ, Σ_range2)
                ΣJ3 .= right_block3 .* view(Σ, Σ_range3)

                row_range = (1 + nu * (k + (j - 1) - 1)):(nu * (k + (j - 1)))
                col_range = (1 + nu * (k - 1)):(nu * k)

                LinearAlgebra.mul!(view(H, row_range, col_range), left_block1', ΣJ1, alpha, 1)
                LinearAlgebra.mul!(view(H, row_range, col_range), left_block2', ΣJ2, alpha, 1)
                LinearAlgebra.mul!(view(H, row_range, col_range), left_block3', ΣJ3, alpha, 1)
            end
        end
    end
end

end # module
