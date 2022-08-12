
"""
SparseLQDynamicModel(dnlp::LQDynamicData)    -> SparseLQDynamicModel
SparseLQDynamicModel(s0, A, B, Q, R, N; ...) -> SparseLQDynamicModel
A constructor for building a `SparseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel`
Input data is for the problem of the form
```math
minimize    \\frac{1}{2} \\sum_{i = 0}^{N-1}(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + \\frac{1}{2} s_N^T Qf s_N
subject to  s_{i+1} = A s_i + B u_i + w for i=0, 1, ..., N-1
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
    w::V  = _init_similar(s0, length(s0), T),
    sl::V = (similar(s0) .= -Inf),
    su::V = (similar(s0) .=  Inf),
    ul::V = (similar(s0, size(R, 1)) .= -Inf),
    uu::V = (similar(s0, size(R, 1)) .=  Inf),
    gl::V = (similar(s0, size(E, 1)) .= -Inf),
    gu::V = (similar(s0, size(F, 1)) .= Inf)
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Union{Nothing, AbstractMatrix{T}}}

    dnlp = LQDynamicData(
        s0, A, B, Q, R, N;
        Qf = Qf, S = S, E = E, F = F, K = K, w = w,
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu
    )

    SparseLQDynamicModel(dnlp)
end


function _build_sparse_lq_dynamic_model(
    dnlp::LQDynamicData{T, V, M, MK}
) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: Nothing}
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
    w  = dnlp.w

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

        lcon[(1 + (i - 1) * ns):(i * ns)] = -w
        ucon[(1 + (i - 1) * ns):(i * ns)] = -w

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

function _build_sparse_lq_dynamic_model(
    dnlp::LQDynamicData{T, V, M, MK}
) where {T, V <: AbstractVector{T}, M  <: AbstractMatrix{T}, MK <: AbstractMatrix}
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
    w  = dnlp.w

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

        lcon[(1 + (i - 1) * ns):(i * ns)] = -w
        ucon[(1 + (i - 1) * ns):(i * ns)] = -w

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

function _build_sparse_lq_dynamic_model(dnlp::LQDynamicData{T, V, M, MK}) where {T, V <: AbstractVector{T}, M  <: SparseMatrixCSC{T}, MK <: Nothing}
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
    w  = dnlp.w

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu

    nc = size(E, 1)

    SparseArrays.dropzeros!(A)
    SparseArrays.dropzeros!(B)
    SparseArrays.dropzeros!(Q)
    SparseArrays.dropzeros!(R)
    SparseArrays.dropzeros!(Qf)
    SparseArrays.dropzeros!(E)
    SparseArrays.dropzeros!(F)
    SparseArrays.dropzeros!(S)

    H_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    H_rowval = zeros(Int, length(Q.rowval) * N + length(R.rowval) * N + 2 * length(S.rowval) * N + length(Qf.rowval))
    H_nzval  = zeros(T, length(Q.nzval) * N + length(R.nzval) * N + 2 * length(S.nzval) * N + length(Qf.nzval))

    J_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    J_rowval = zeros(Int, length(A.rowval) * N + length(B.rowval) * N + length(E.rowval) * N + length(F.rowval) * N + ns * N)
    J_nzval  = zeros(T, length(A.nzval) * N + length(B.nzval) * N + length(E.nzval) * N + length(F.nzval) * N + ns * N)

    _set_sparse_H!(H_colptr, H_rowval, H_nzval, Q, R, N; Qf = Qf, S = S)

    H = SparseArrays.SparseMatrixCSC((N + 1) * ns + nu * N, (N + 1) * ns + nu * N, H_colptr, H_rowval, H_nzval)

    _set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, N)

    J = SparseArrays.SparseMatrixCSC((nc + ns) * N, (N + 1) * ns + nu * N, J_colptr, J_rowval, J_nzval)

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

        lcon[(1 + (i - 1) * ns):(i * ns)] = -w
        ucon[(1 + (i - 1) * ns):(i * ns)] = -w

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


function _build_sparse_lq_dynamic_model(dnlp::LQDynamicData{T, V, M, MK}) where {T, V <: AbstractVector{T}, M  <: SparseMatrixCSC{T}, MK <: SparseMatrixCSC{T}}
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
    w  = dnlp.w

    sl = dnlp.sl
    su = dnlp.su
    ul = dnlp.ul
    uu = dnlp.uu
    gl = dnlp.gl
    gu = dnlp.gu

    nc = size(E, 1)

    SparseArrays.dropzeros!(A)
    SparseArrays.dropzeros!(B)
    SparseArrays.dropzeros!(Q)
    SparseArrays.dropzeros!(R)
    SparseArrays.dropzeros!(Qf)
    SparseArrays.dropzeros!(E)
    SparseArrays.dropzeros!(F)
    SparseArrays.dropzeros!(S)
    SparseArrays.dropzeros!(K)

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
    H_rowval = zeros(Int, length(Q.rowval) * N + length(R.rowval) * N + 2 * length(S.rowval) * N + length(Qf.rowval))
    H_nzval  = zeros(T, length(Q.nzval) * N + length(R.nzval) * N + 2 * length(S.nzval) * N + length(Qf.nzval))

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

    SparseArrays.dropzeros!(new_Q)
    SparseArrays.dropzeros!(new_A)
    SparseArrays.dropzeros!(new_E)
    SparseArrays.dropzeros!(new_S)

    K_sparse = K[bool_vec, :]

    H_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    H_rowval = zeros(Int, length(Q.rowval) * N + length(R.rowval) * N + 2 * length(new_S.rowval) * N + length(Qf.rowval))
    H_nzval  = zeros(T, length(Q.nzval) * N + length(R.nzval) * N + 2 * length(new_S.nzval) * N + length(Qf.nzval))


    J_colptr = zeros(Int, ns * (N + 1) + nu * N + 1)
    J_rowval = zeros(Int, length(new_A.rowval) * N + length(B.rowval) * N + length(new_E.rowval) * N + length(F.rowval) * N + ns * N + length(K_sparse.rowval) * N + num_real_bounds * N)
    J_nzval  = zeros(T, length(new_A.nzval) * N + length(B.nzval) * N + length(new_E.nzval) * N + length(F.nzval) * N + ns * N + length(K_sparse.nzval) * N + num_real_bounds * N)

    # Get H and J matrices from new matrices
    _set_sparse_H!(H_colptr, H_rowval, H_nzval, new_Q, R, N; Qf = Qf, S = new_S)

    H = SparseArrays.SparseMatrixCSC((N + 1) * ns + nu * N, (N + 1) * ns + nu * N, H_colptr, H_rowval, H_nzval)

    _set_sparse_J!(J_colptr, J_rowval, J_nzval, new_A, B, new_E, F, K, bool_vec, N, num_real_bounds)

    J = SparseArrays.SparseMatrixCSC(ns * N + nc * N + num_real_bounds * N, (N + 1) * ns + nu * N, J_colptr, J_rowval, J_nzval)

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

        lcon[(1 + (i - 1) * ns):(i * ns)] = -w
        ucon[(1 + (i - 1) * ns):(i * ns)] = -w

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

function _set_sparse_H!(
    H_colptr, H_rowval, H_nzval,
    Q::M, R::M, N;
    Qf::M = Q,
    S::M = spzeros(T, size(Q, 1), size(R, 1))
) where {T, M <: SparseMatrixCSC{T}}

    ST = SparseArrays.sparse(S')
    ns = size(Q, 1)
    nu = size(R, 1)

    H_colptr[1] = 1

    for i in 1:N
        for j in 1:ns
            Q_offset = length(Q.colptr[j]:(Q.colptr[j + 1] - 1))

            H_nzval[(H_colptr[ns * (i - 1) + j]):(H_colptr[ns * (i - 1) + j] + Q_offset - 1)]  = Q.nzval[Q.colptr[j]:(Q.colptr[j + 1] - 1)]
            H_rowval[(H_colptr[ns * (i - 1) + j]):(H_colptr[ns * (i - 1) + j] + Q_offset - 1)] = Q.rowval[Q.colptr[j]:(Q.colptr[j + 1] - 1)] .+ ns * (i - 1)

            ST_offset = length(ST.colptr[j]:(ST.colptr[j + 1] - 1))
            H_nzval[(H_colptr[ns * (i - 1) + j] + Q_offset):(H_colptr[ns * (i - 1) + j] + Q_offset + ST_offset - 1)]  = ST.nzval[ST.colptr[j]:(ST.colptr[j + 1] - 1)]
            H_rowval[(H_colptr[ns * (i - 1) + j] + Q_offset):(H_colptr[ns * (i - 1) + j] + Q_offset + ST_offset - 1)] = ST.rowval[ST.colptr[j]:(ST.colptr[j + 1] - 1)] .+ (nu * (i - 1) + ns * (N + 1))

            H_colptr[ns * (i - 1) + j + 1] = H_colptr[ns * (i - 1) + j] + Q_offset + ST_offset
        end
    end

    for j in 1:ns
        Qf_offset = length(Qf.colptr[j]:(Qf.colptr[j + 1] - 1))
        H_nzval[(H_colptr[N * ns + j]):(H_colptr[N * ns + j] + Qf_offset - 1)]  = Qf.nzval[Qf.colptr[j]:(Qf.colptr[j + 1] - 1)]
        H_rowval[(H_colptr[N * ns + j]):(H_colptr[N * ns + j] + Qf_offset - 1)] = Qf.rowval[Qf.colptr[j]:(Qf.colptr[j + 1] - 1)] .+ (ns * N)
        H_colptr[ns * N + j + 1] = H_colptr[ns * N + j] + Qf_offset
    end

    for i in 1:N
        for j in 1:nu
            S_offset = length(S.colptr[j]:(S.colptr[j + 1] - 1))

            H_nzval[(H_colptr[ns * (N + 1) + nu * (i - 1) + j]):(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset - 1)]  = S.nzval[S.colptr[j]:(S.colptr[j + 1] - 1)]
            H_rowval[(H_colptr[ns * (N + 1) + nu * (i - 1) + j]):(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset - 1)] = S.rowval[S.colptr[j]:(S.colptr[j + 1] - 1)] .+ ((i - 1) * ns)

            R_offset = length(R.colptr[j]:(R.colptr[j + 1] - 1))

            H_nzval[(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset):(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset + R_offset - 1)]  = R.nzval[R.colptr[j]:(R.colptr[j + 1] - 1)]
            H_rowval[(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset):(H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset + R_offset - 1)] = R.rowval[R.colptr[j]:(R.colptr[j + 1] - 1)] .+ ((i - 1) * nu + ns * (N + 1))

            H_colptr[ns * (N + 1) + nu * (i - 1) + j + 1] = H_colptr[ns * (N + 1) + nu * (i - 1) + j] + S_offset + R_offset
        end
    end
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

function _set_sparse_J!(
    J_colptr, J_rowval, J_nzval,
    A::M, B::M, E::M, F::M, K::MK, bool_vec,
    N, nb
) where {T, M <: SparseMatrixCSC{T}, MK <: SparseMatrixCSC{T}}

    ns = size(A, 2)
    nu = size(B, 2)
    nc = size(E, 1)

    I_mat = _init_similar(K, nu, nu)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    KI       = I_mat[bool_vec, :]
    K_sparse = K[bool_vec, :]

    J_colptr[1] = 1

    # Set the first block column of A, E, and K
    for j in 1:ns
        A_offset = length(A.colptr[j]:(A.colptr[j + 1] - 1))
        J_nzval[(J_colptr[j]):(J_colptr[j] + A_offset - 1)]  = A.nzval[A.colptr[j]:(A.colptr[j + 1] - 1)]
        J_rowval[(J_colptr[j]):(J_colptr[j] + A_offset - 1)] = A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)]

        E_offset = length(E.colptr[j]:(E.colptr[j + 1] - 1))

        J_nzval[(J_colptr[j] + A_offset):(J_colptr[j] + A_offset + E_offset - 1)]  = E.nzval[E.colptr[j]:(E.colptr[j + 1] - 1)]
        J_rowval[(J_colptr[j] + A_offset):(J_colptr[j] + A_offset + E_offset - 1)] = E.rowval[E.colptr[j]:(E.colptr[j + 1] - 1)] .+ (ns * N)

        K_offset = length(K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1))

        (J_nzval[(J_colptr[j] + A_offset + E_offset):(J_colptr[j] + A_offset + E_offset + K_offset - 1)]
            = K_sparse.nzval[K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1)])
        (J_rowval[(J_colptr[j] + A_offset + E_offset):(J_colptr[j] + A_offset + E_offset + K_offset - 1)]
            = K_sparse.rowval[K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1)] .+ ((ns + nc) * N))

        J_colptr[j + 1] = J_colptr[j] + A_offset + E_offset + K_offset
    end

    # Set the remaining block columns corresponding to states: -I, A, E, K
    for i in 2:N
        for j in 1:ns
            J_nzval[J_colptr[j + (i - 1) * ns]]  = T(-1)
            J_rowval[J_colptr[j + (i - 1) * ns]] = ns * (i - 2) + j

            A_offset = length(A.colptr[j]:(A.colptr[j + 1] - 1))

            J_nzval[(J_colptr[j + (i - 1) * ns] + 1):(J_colptr[j + (i - 1) * ns] + 1 + A_offset - 1)]  = A.nzval[A.colptr[j]:(A.colptr[j + 1] - 1)]
            J_rowval[(J_colptr[j + (i - 1) * ns] + 1):(J_colptr[j + (i - 1) * ns] + 1 + A_offset - 1)] = A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)] .+ (ns * (i - 1))

            E_offset = length(E.colptr[j]:(E.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset - 1)]
                = E.nzval[E.colptr[j]:(E.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset - 1)]
                = E.rowval[E.colptr[j]:(E.colptr[j + 1] - 1)] .+ (ns * N + nc * (i - 1)))

            K_offset = length(K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset + K_offset - 1)]
                = K_sparse.nzval[K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset + K_offset - 1)]
                = K_sparse.rowval[K_sparse.colptr[j]:(K_sparse.colptr[j + 1] - 1)] .+ ((ns + nc) * N + nb * (i - 1)))

            J_colptr[ns * (i - 1) + j + 1] = J_colptr[ns * (i - 1) + j] + 1 + A_offset + E_offset + K_offset
        end
    end

    # Set the column corresponding to states at N + 1, which are a single block of -I
    for j in 1:ns
        J_nzval[J_colptr[ns * N + j]]  = T(-1)
        J_rowval[J_colptr[ns * N + j]] = ns * (N - 1) + j
        J_colptr[ns * N + j + 1] = J_colptr[ns * N + j] + 1
    end

    # Set the remaining block columns corresponding to inputs: B, F, I
    for i in 1:N
        offset = ns * (N + 1) + nu * (i - 1)
        for j in 1:nu
            B_offset = length(B.colptr[j]:(B.colptr[j + 1] - 1))

            J_nzval[(J_colptr[offset + j]):(J_colptr[offset + j] + B_offset - 1)]  = B.nzval[B.colptr[j]:(B.colptr[j + 1] - 1)]
            J_rowval[(J_colptr[offset + j]):(J_colptr[offset + j] + B_offset - 1)] = B.rowval[B.colptr[j]:(B.colptr[j + 1] - 1)] .+ (ns * (i -1))

            F_offset = length(F.colptr[j]:(F.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[offset + j] + B_offset):(J_colptr[offset + j] + B_offset + F_offset - 1)]
                = F.nzval[F.colptr[j]:(F.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[offset + j] + B_offset):(J_colptr[offset + j] + B_offset + F_offset - 1)]
                = F.rowval[F.colptr[j]:(F.colptr[j + 1] - 1)] .+ (ns * N + nc * (i - 1)))

            KI_offset = length(KI.colptr[j]:(KI.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[offset + j] + B_offset + F_offset):(J_colptr[offset + j] + B_offset + F_offset + KI_offset - 1)]
                = KI.nzval[KI.colptr[j]:(KI.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[offset + j] + B_offset + F_offset):(J_colptr[offset + j] + B_offset + F_offset + KI_offset - 1)]
                = KI.rowval[KI.colptr[j]:(KI.colptr[j + 1] - 1)] .+ ((ns + nc) * N + nb * (i - 1)))

            J_colptr[offset + j + 1] = J_colptr[offset + j] + B_offset + F_offset + KI_offset
        end
    end
end

function _set_sparse_J!(
    J_colptr, J_rowval, J_nzval,
    A::M, B::M, E::M, F::M, K::MK, N
) where {T, M <: SparseMatrixCSC{T}, MK <: Nothing}

    ns = size(A, 2)
    nu = size(B, 2)
    nc = size(E, 1)

    J_colptr[1] = 1
    # Set the first block column of A, E, and K
    for j in 1:ns
        A_offset = length(A.colptr[j]:(A.colptr[j + 1] - 1))
        J_nzval[(J_colptr[j]):(J_colptr[j] + A_offset - 1)]  = A.nzval[A.colptr[j]:(A.colptr[j + 1] - 1)]
        J_rowval[(J_colptr[j]):(J_colptr[j] + A_offset - 1)] = A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)]

        E_offset = length(E.colptr[j]:(E.colptr[j + 1] - 1))

        J_nzval[(J_colptr[j] + A_offset):(J_colptr[j] + A_offset + E_offset - 1)]  = E.nzval[E.colptr[j]:(E.colptr[j + 1] - 1)]
        J_rowval[(J_colptr[j] + A_offset):(J_colptr[j] + A_offset + E_offset - 1)] = E.rowval[E.colptr[j]:(E.colptr[j + 1] - 1)] .+ (ns * N)

        J_colptr[j + 1] = J_colptr[j] + A_offset + E_offset
    end

    # Set the remaining block columns corresponding to states: -I, A, E
    for i in 2:N
        for j in 1:ns
            J_nzval[J_colptr[j + (i - 1) * ns]]  = T(-1)
            J_rowval[J_colptr[j + (i - 1) * ns]] = ns * (i - 2) + j

            A_offset = length(A.colptr[j]:(A.colptr[j + 1] - 1))

            J_nzval[(J_colptr[j + (i - 1) * ns] + 1):(J_colptr[j + (i - 1) * ns] + 1 + A_offset - 1)]  = A.nzval[A.colptr[j]:(A.colptr[j + 1] - 1)]
            J_rowval[(J_colptr[j + (i - 1) * ns] + 1):(J_colptr[j + (i - 1) * ns] + 1 + A_offset - 1)] = A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)] .+ (ns * (i - 1))

            E_offset = length(E.colptr[j]:(E.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset - 1)]
                = E.nzval[E.colptr[j]:(E.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[j + (i - 1) * ns] + 1 + A_offset):(J_colptr[j + (i - 1) * ns] + 1 + A_offset + E_offset - 1)]
                = E.rowval[E.colptr[j]:(E.colptr[j + 1] - 1)] .+ (ns * N + nc * (i - 1)))

            J_colptr[ns * (i - 1) + j + 1] = J_colptr[ns * (i - 1) + j] + 1 + A_offset + E_offset
        end
    end

    # Set the column corresponding to states at N + 1, which are a single block of -I
    for j in 1:ns
        J_nzval[J_colptr[ns * N + j]]  = T(-1)
        J_rowval[J_colptr[ns * N + j]] = ns * (N - 1) + j
        J_colptr[ns * N + j + 1] = J_colptr[ns * N + j] + 1
    end

    # Set the remaining block columns corresponding to inputs: B, F
    for i in 1:N
        offset = ns * (N + 1) + nu * (i - 1)
        for j in 1:nu
            B_offset = length(B.colptr[j]:(B.colptr[j + 1] - 1))

            J_nzval[(J_colptr[offset + j]):(J_colptr[offset + j] + B_offset - 1)]  = B.nzval[B.colptr[j]:(B.colptr[j + 1] - 1)]
            J_rowval[(J_colptr[offset + j]):(J_colptr[offset + j] + B_offset - 1)] = B.rowval[B.colptr[j]:(B.colptr[j + 1] - 1)] .+ (ns * (i -1))

            F_offset = length(F.colptr[j]:(F.colptr[j + 1] - 1))

            (J_nzval[(J_colptr[offset + j] + B_offset):(J_colptr[offset + j] + B_offset + F_offset - 1)]
                = F.nzval[F.colptr[j]:(F.colptr[j + 1] - 1)])
            (J_rowval[(J_colptr[offset + j] + B_offset):(J_colptr[offset + j] + B_offset + F_offset - 1)]
                = F.rowval[F.colptr[j]:(F.colptr[j + 1] - 1)] .+ (ns * N + nc * (i - 1)))

            J_colptr[offset + j + 1] = J_colptr[offset + j] + B_offset + F_offset
        end
    end
end
