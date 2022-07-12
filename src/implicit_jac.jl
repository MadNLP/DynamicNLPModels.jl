mutable struct JacobianOperator{T, V, M} <: LinearOperators.LinearOperator{T}
    Jac::M   # column of Jacobian block matrix
    N        # number of time steps
    nc       # number of inequality constraints
    nsc      # number of state variables that are constrained
    nuc      # number of input variables that are constrained

    scaled_Jac::M

    # Storage vectors for J x
    J1Bx::V
    J2Bx::V
    J3Bx::V

    # Storage vectors for J^T x
    J1BTx::V
    J2BTx::V
    J3BTx::V

    # Storage matices for building J^TΣJ
    J1B::M
    J2B::M
    J3B::M
end


function _build_truncated_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Nothing}
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
    Jac  = _init_similar(Q, nc * N + num_real_bounds_s * N, nu, T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    Jac[1:nc * N, :] = G

    As0 = _init_similar(s0, ns * (N + 1), T)
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

    lcon = _init_similar(s0, length(dl) + length(lcon2), T)
    ucon = _init_similar(s0, length(du) + length(ucon2), T)

    lcon[1:length(dl)] = dl
    ucon[1:length(du)] = du

    if length(lcon2) > 0
        lcon[(1 + length(dl)):(length(dl) + num_real_bounds_s * N)] = lcon2
        ucon[(1 + length(du)):(length(du) + num_real_bounds_s * N)] = ucon2
    end

    nvar = nu * N
    nnzj = size(J, 1) * size(H, 2)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(J, 1)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    J = JacobianOperator{T, V, M}(
        Jac, N, nc, num_real_bounds_s, num_real_bounds_u,
        Jac,
        _init_similar(s0, nc, T), _init_similar(s0, num_real_bounds_s, T), _init_similar(s0, 0, T),
        _init_similar(s0, nc, T), _init_similar(s0, num_real_bounds_s, T), _init_similar(s0, 0, T),
        _init_similar(s0, nc, nu, T), _init_similar(s0, num_real_bounds_s, nu, T), _init_similar(s0, 0, nu, T),
    )

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

function _build_truncated_dense_lq_dynamic_model(dnlp::LQDynamicData{T,V,M,MK}) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}
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

    I_mat = _init_similar(Q, nu, nu, T)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, s0, E, F, K, N)
    Jac  = _init_similar(Q, nc * N + num_real_bounds_s * N, nu, T)

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

            Jac[(1 + offset_u + (i - 1) * num_real_bounds_u):offset_s + i * num_real_bounds_u] = @view KBI[(1 + (i - 1) * nu):(i * nu), :][bool_vec_u, :]
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
    nnzj = size(J, 1) * size(H, 1)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)
    ncon = size(J, 1)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    J = JacobianOperator{T, V, M}(
        Jac, N, nc, num_real_bounds_s, num_real_bounds_u,
        Jac,
        _init_similar(s0, nc, T), _init_similar(s0, num_real_bounds_s, T), _init_similar(s0, num_real_bounds_u, T),
        _init_similar(s0, nu, T), _init_similar(s0, nu, T), _init_similar(s0, nu, T),
        _init_similar(s0, nc, nu, T), _init_similar(s0, num_real_bounds_s, nu, T), _init_similar(s0, num_real_bounds_u, nu, T),
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
