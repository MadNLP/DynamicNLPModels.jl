abstract type AbstractLQDynData{T, V} end
@doc raw"""
    LQDynamicData{T,V,M,MK} <: AbstractLQDynData{T,V}

A struct to represent the features of the optimization problem

```math
    \begin{aligned}
        \min \frac{1}{2} &\; \sum_{i = 0}^{N - 1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \frac{1}{2} s_N^T Q_f s_N \\
        \textrm{s.t.} &\; s_{i+1} = A s_i + B u_i + w_i  \quad \forall i=0, 1, ..., N-1 \\
        &\; u_i = Kx_i + v_i \quad  \forall i = 0, 1, ..., N - 1 \\
        &\; g^l \le E s_i + F u_i \le g^u \quad \forall i = 0, 1, ..., N-1\\
        &\; s^l \le s \le s^u \\
        &\; u^l \le u \le u^u \\
        &\; s_0 = s0
    \end{aligned}
```

---
Attributes include:
- `s0`: initial state of system
- `A` : constraint matrix for system states
- `B` : constraint matrix for system inputs
- `Q` : objective function matrix for system states from 0:(N-1)
- `R` : objective function matrix for system inputs from 0:(N-1)
- `N` : number of time steps
- `Qf`: objective function matrix for system state at time N
- `S` : objective function matrix for system states and inputs
- `ns`: number of state variables
- `nu`: number of input varaibles
- `E` : constraint matrix for state variables
- `F` : constraint matrix for input variables
- `K` : feedback gain matrix
- 'w' : constant term for dynamic constraints
- `sl`: vector of lower bounds on state variables
- `su`: vector of upper bounds on state variables
- `ul`: vector of lower bounds on input variables
- `uu`: vector of upper bounds on input variables
- `gl`: vector of lower bounds on constraints
- `gu`: vector of upper bounds on constraints

see also `LQDynamicData(s0, A, B, Q, R, N; ...)`
"""
struct LQDynamicData{T, V, M, MK} <: AbstractLQDynData{T, V}
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
    w::V

    sl::V
    su::V
    ul::V
    uu::V
    gl::V
    gu::V
end

@doc raw"""
    LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, V, M, MK}
A constructor for building an object of type `LQDynamicData` for the optimization problem
```math
    \begin{aligned}
        \min \frac{1}{2} &\; \sum_{i = 0}^{N - 1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \frac{1}{2} s_N^T Q_f s_N \\
        \textrm{s.t.} &\; s_{i+1} = A s_i + B u_i + w_i  \quad \forall i=0, 1, ..., N-1 \\
        &\; u_i = Kx_i + v_i \quad  \forall i = 0, 1, ..., N - 1 \\
        &\; gl \le E s_i + F u_i \le gu \quad \forall i = 0, 1, ..., N-1\\
        &\; sl \le s \le su \\
        &\; ul \le u \le uu \\
        &\; s_0 = s0
    \end{aligned}
```
---
- `s0`: initial state of system
- `A` : constraint matrix for system states
- `B` : constraint matrix for system inputs
- `Q` : objective function matrix for system states from 0:(N-1)
- `R` : objective function matrix for system inputs from 0:(N-1)
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
- `w  = zeros(eltype(Q), ns * N)`     : constant term for dynamic constraints
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
    S::M = _init_similar(Q, size(Q, 1), size(R, 1), T),
    E::M = _init_similar(Q, 0, length(s0), T),
    F::M = _init_similar(Q, 0, size(R, 1), T),
    K::MK = nothing,
    w::V = _init_similar(s0, length(s0) * N, T),
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .= Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .= Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf),
    gu::V = (similar(s0, size(F, 1)) .= Inf),
) where {
    T,
    V <: AbstractVector{T},
    M <: AbstractMatrix{T},
    MK <: Union{Nothing, AbstractMatrix{T}},
}

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

    if !all(sl .<= su)
        error("lower bound(s) on s is > upper bound(s)")
    end
    if !all(ul .<= uu)
        error("lower bound(s) on u is > upper bound(s)")
    end
    if !all(sl .<= s0) || !all(s0 .<= su)
        error("s0 is not within the given upper and lower bounds")
    end

    if size(E, 1) != size(F, 1)
        error("E and F have different numbers of rows")
    end
    if !all(gl .<= gu)
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
        if size(K, 1) != size(R, 1) || size(K, 2) != size(Q, 1)
            error("Dimensions of K  do not match number of states and inputs")
        end
    end
    if Int(size(w, 1)) != Int(size(s0, 1) * N)
        error("Dimensions of w do not match ns")
    end


    ns = size(Q, 1)
    nu = size(R, 1)

    LQDynamicData{T, V, M, MK}(
        s0,
        A,
        B,
        Q,
        R,
        N,
        Qf,
        S,
        ns,
        nu,
        E,
        F,
        K,
        w,
        sl,
        su,
        ul,
        uu,
        gl,
        gu,
    )
end

abstract type AbstractDynamicModel{T, V} <: QuadraticModels.AbstractQuadraticModel{T, V} end

struct SparseLQDynamicModel{T, V, M1, M2, M3, MK} <: AbstractDynamicModel{T, V}
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    data::QuadraticModels.QPData{T, V, M1, M2}
    dynamic_data::LQDynamicData{T, V, M3, MK}
end

"""
Struct containing block matrices used for creating and resetting the `DenseLQDynamicModel`. A and B matrices are given in part by
Jerez, Kerrigan, and Constantinides in section 4 of "A sparse and condensed QP formulation for predictive control of LTI systems"
(doi:10.1016/j.automatica.2012.03.010). States are eliminated by the equation  ``x = Ax_0 + Bu + \\hat{A}w`` where  ``x = [x_0^T, x_1^T, ..., x_N^T]``
and ``u = [u_0^T, u_1^T, ..., u_{N-1}^T]``

---
- `A`  : block A matrix given by Jerez et al. with ``n_s(N + 1)`` rows and ns columns
- `B`  : block B matrix given by Jerez et al. with ``n_s(N)`` rows and nu columns
- `Aw` : length ``n_s(N + 1)`` vector corresponding to the linear term of the dynamic constraints
- `h`  : ``n_u(N) \\times n_s`` matrix for building the linear term of the objective function. Just needs to be
multiplied by `s0`.
- `h01`: ns x ns matrix for building the constant term fo the objective function. This can be found by
taking  ``s_0^T``  `h01`  ``s_0``
- `h02`: similar to `h01`, but one side is multiplied by `Aw` rather than by `As0`. This will just
be multiplied by `s0` once
- `h_constant` : linear term in the objective function that arises from `Aw`. Not a function of `s0`
- `h0_constant`: constant term in the objective function that arises from `Aw`. Not a function of `s0`
- `d`  : length ``n_c(N)`` term for the constraint bounds corresponding to `E` and `F`. Must be multiplied by `s0` and
subtracted from `gl` and `gu`. Equal to the blocks (E + FK) A (see Jerez et al.)
- `dw` : length ``n_c(N)`` term for the constraint bounds that arises from `w`. Equal to the blocks (E + FK) Aw
- `KA` : size ``n_u(N)`` x ns matrix. Needs to be multiplied by `s0` and subtracted from `ul` and `uu` to update
the algebraic constraints corresponding to the input bounds
- `KAw`: similar to `KA`, but it is multiplied by Aw rather than A

See also `reset_s0!`
"""
mutable struct DenseLQDynamicBlocks{T, V, M}
    A::M
    B::M
    Aw::V          # Aw = block_matrix_A * w (result is a Vector; block_matrix A is like block_B, but with I instead of B)
    h::M           # h = (QB + SKB + K^T R K B + K^T S^T B)^T A + (S + K^T R)^T A
    h01::M         # h01 = A^T((Q + KTRK + KTST + SK))A where Q, K, R, S, and A are block matrices just needs to be multiplied by s0 on each side
    h02::V         # h02 = wT block_matrix_AT (Q + KTRK + KTSK + SK) A; just needs to be multiplied by s0 on right
    h_constant::V  # h_constant  = BT (Q + KTRK + SK + KTST) block_matrix_A w + (RK + ST)B block_matrix_A w
    h0_constant::T # h0_constant = wT block_matrix_AT (Q + KTRK + KTSK + SK) block_matrix_A w
    d::M           # d = (E + FK) A
    dw::V          # dw = (E + FK) block_matrix_A w - constant term to be subtracted from d
    KA::M
    KAw::V
end

struct DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK} <: AbstractDynamicModel{T, V}
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    data::QuadraticModels.QPData{T, V, M1, M2}
    dynamic_data::LQDynamicData{T, V, M3, MK}
    blocks::DenseLQDynamicBlocks{T, V, M4}
end

"""
    LQJacobianOperator{T, V, M}

Struct for storing the implicit Jacobian matrix. All data for the Jacobian can be stored
in the first `nu` columns of `J`. This struct contains the needed data and storage arrays for
calculating ``Jx``, ``J^T x``, and ``J^T \\Sigma J``. ``Jx`` and ``J^T x`` are performed through extensions
to `LinearAlgebra.mul!()`.

---
Attributes
 - `truncated_jac1`: Matrix of first `nu` columns of the Jacobian corresponding to Ax + Bu constraints
 - `truncated_jac2`: Matrix of first `nu` columns of the Jacobian corresponding to state variable bounds
 - `truncated_jac3`: Matrix of first `nu` columns of the Jacobian corresponding to input variable bounds
 - `N`  : number of time steps
 - `nu` : number of inputs
 - `nc` : number of algebraic constraints of the form gl <= Es + Fu <= gu
 - `nsc`: number of bounded state variables
 - `nuc`: number of bounded input variables (if `K` is defined)
 - `SJ1`: placeholder for storing data when calculating `ΣJ`
 - `SJ2`: placeholder for storing data when calculating `ΣJ`
 - `SJ3`: placeholder for storing data when calculating `ΣJ`
 - `H_sub_block`: placeholder for storing data when adding `J^T ΣJ` to the Hessian
"""
struct LQJacobianOperator{T, M, A} <: LinearOperators.AbstractLinearOperator{T}
    truncated_jac1::A  # tensor of Jacobian blocks corresponding Ex + Fu constraints
    truncated_jac2::A  # tensor of Jacobian blocks corresponding to state variable limits
    truncated_jac3::A  # tensor of Jacobian blocks corresponding to input variable limits

    N::Int             # number of time steps
    nu::Int            # number of inputs
    nc::Int            # number of inequality constraints
    nsc::Int           # number of state variables that are constrained
    nuc::Int           # number of input variables that are constrained

    # Storage tensors for building Jx and J^Tx
    x1::A
    x2::A
    x3::A
    y::A

    # Storage tensors for building J^TΣJ
    SJ1::M
    SJ2::M
    SJ3::M

    # Storage block for adding J^TΣJ to H
    H_sub_block::M
end


function _init_similar(mat, dim1::Number, dim2::Number, dim3::Number, T::DataType)
    new_mat = similar(mat, dim1, dim2, dim3)
    fill!(new_mat, zero(T))
    return new_mat
end

function _init_similar(mat, dim1::Number, dim2::Number, T = eltype(mat))
    new_mat = similar(mat, dim1, dim2)
    fill!(new_mat, zero(T))
    return new_mat
end

function _init_similar(mat, dim1::Number, T = eltype(mat))
    new_mat = similar(mat, dim1)
    fill!(new_mat, zero(T))
    return new_mat
end
