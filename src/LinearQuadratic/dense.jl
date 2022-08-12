"""
DenseLQDynamicModel(dnlp::LQDynamicData; implicit = false)    -> DenseLQDynamicModel
DenseLQDynamicModel(s0, A, B, Q, R, N; implicit = false ...) -> DenseLQDynamicModel
A constructor for building a `DenseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel`

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
    w::V  = _init_similar(s0, length(s0), T),
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
        Qf = Qf, S = S, E = E, F = F, K = K, w = w,
        sl = sl, su = su, ul = ul, uu = uu, gl = gl, gu = gu
    )

    DenseLQDynamicModel(dnlp; implicit = implicit)
end


function _build_dense_lq_dynamic_model(
    dnlp::LQDynamicData{T,V,M,MK}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Nothing}
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

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s = sum(bool_vec_s)

    dense_blocks = _build_block_matrices(A, B, K, N, w, nc)
    block_A  = dense_blocks.A
    block_B  = dense_blocks.B
    block_d  = dense_blocks.d
    block_Aw = dense_blocks.Aw
    block_dw = dense_blocks.dw

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, block_Aw, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0
    dense_blocks.h   .= H_blocks.block_h
    dense_blocks.h01 .= H_blocks.block_h01
    dense_blocks.h02 .= H_blocks.block_h02
    dense_blocks.h_constant  .= H_blocks.h_constant
    dense_blocks.h0_constant = H_blocks.h0_constant

    G  = _init_similar(Q, nc * N, nu, T)
    J  = _init_similar(Q, nc * N + num_real_bounds_s * N, nu * N, T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, block_d, block_Aw, block_dw, s0, E, F, K, N)
    _set_J1_dense!(J, G, N)

    As0 = _init_similar(s0, ns * (N + 1), T)
    LinearAlgebra.mul!(As0, block_A, s0)

    lvar = repeat(ul, N)
    uvar = repeat(uu, N)

    # Convert state variable constraints to algebraic constraints
    offset_s = N * nc
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)] .+ block_Aw[(1 + ns):(ns * (N + 1))]
        for i in 1:N
            J[(offset_s + 1 + (i - 1) * ns):(offset_s + ns * N), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[1:(ns * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .+ block_Aw[(1 + ns * i):(ns * (i + 1))][bool_vec_s]

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

function _build_dense_lq_dynamic_model(
    dnlp::LQDynamicData{T,V,M,MK}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}
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

    dense_blocks = _build_block_matrices(A, B, K, N, w, nc)
    block_A   = dense_blocks.A
    block_B   = dense_blocks.B
    block_d   = dense_blocks.d
    block_Aw  = dense_blocks.Aw
    block_dw  = dense_blocks.dw
    block_KAw = dense_blocks.KAw

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, block_Aw, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0
    dense_blocks.h   .= H_blocks.block_h
    dense_blocks.h01 .= H_blocks.block_h01
    dense_blocks.h02 .= H_blocks.block_h02
    dense_blocks.h_constant  .= H_blocks.h_constant
    dense_blocks.h0_constant = H_blocks.h0_constant

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s   = sum(bool_vec_s)

    bool_vec_u       = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds_u  = sum(bool_vec_u)


    G   = _init_similar(Q, nc * N, nu, T)
    J   = _init_similar(Q, (nc + num_real_bounds_s + num_real_bounds_u) * N, nu * N, T)
    As0 = _init_similar(s0, ns * (N + 1), T)
    As0_bounds  = _init_similar(s0, num_real_bounds_s * N, T)
    KAs0_bounds = _init_similar(s0, num_real_bounds_u * N, T)

    KBI        = _init_similar(Q, nu * N, nu, T)
    KAs0       = _init_similar(s0, nu * N, T)
    KAs0_block = _init_similar(s0, nu, T)
    KB         = _init_similar(Q, nu, nu, T)

    I_mat = _init_similar(Q, nu, nu, T)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, block_d, block_Aw, block_dw, s0, E, F, K, N)
    _set_J1_dense!(J, G, N)

    LinearAlgebra.mul!(As0, block_A, s0)

    # Convert state variable constraints to algebraic constraints
    offset_s = nc * N
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)] .+ block_Aw[(1 + ns):ns * (N + 1)]
        for i in 1:N
            J[(offset_s + 1 + (i - 1) * ns):(offset_s + ns * N), (1 + nu * (i - 1)):(nu * i)] = @view(block_B[1:(ns * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] = As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .+ block_Aw[(1 + ns * i):(ns * (i + 1))][bool_vec_s]

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
        KAs0_bounds .= KAs0 .+ block_KAw
        for i in 1:N
            J[(offset_u + 1 + (i - 1) * nu):(offset_u + nu * N), (1 + nu * (i - 1)):(nu * i)] = @view(KBI[1:(nu * (N - i + 1)),:])
        end
    else
        for i in 1:N
            row_range              = (1 + (i - 1) * num_real_bounds_u):(i * num_real_bounds_u)
            KAs0_bounds[row_range] = KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u] .+ block_KAw[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]

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

function _build_implicit_dense_lq_dynamic_model(
    dnlp::LQDynamicData{T,V,M,MK}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: Nothing}
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
    nvar = nu * N

    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s = sum(bool_vec_s)

    G          = _init_similar(Q, nc * N, nu, T)
    Jac1       = _init_similar(Q, nc, nu, N, T)
    Jac2       = _init_similar(Q, num_real_bounds_s, nu, N, T)
    Jac3       = _init_similar(Q, 0, nu, N, T)

    As0        = _init_similar(s0, ns * (N + 1), T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)

    c  = _init_similar(s0, nvar, T)
    x0 = _init_similar(s0, nvar, T)

    lcon = _init_similar(s0, nc * N + num_real_bounds_s * N, T)
    ucon = _init_similar(s0, nc * N + num_real_bounds_s * N, T)

    x1  = _init_similar(Q, nc, 1, N, T)
    x2  = _init_similar(Q, num_real_bounds_s, 1, N, T)
    x3  = _init_similar(Q, 0, 1, N, T)
    y   = _init_similar(Q, nu, 1, N, T)

    SJ1  = _init_similar(Q, nc, nu, T)
    SJ2  = _init_similar(Q, num_real_bounds_s, nu, T)
    SJ3  = _init_similar(Q, 0, nu, T)
    H_sub_block = _init_similar(Q, nu, nu, T)

    dense_blocks = _build_block_matrices(A, B, K, N, w, nc)
    block_A      = dense_blocks.A
    block_B      = dense_blocks.B
    block_d      = dense_blocks.d
    block_Aw = dense_blocks.Aw
    block_dw = dense_blocks.dw

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, block_Aw, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0
    dense_blocks.h   .= H_blocks.block_h
    dense_blocks.h01 .= H_blocks.block_h01
    dense_blocks.h02 .= H_blocks.block_h02
    dense_blocks.h_constant  .= H_blocks.h_constant
    dense_blocks.h0_constant = H_blocks.h0_constant

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, block_d, block_Aw, block_dw, s0, E, F, K, N)
    for i in 1:N
        Jac1[:, :, i] = @view G[(1 + nc * (i - 1)):(nc * i), :]
    end

    LinearAlgebra.mul!(As0, block_A, s0)

    lvar = repeat(ul, N)
    uvar = repeat(uu, N)

    # Convert state variable constraints to algebraic constraints
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)] .+ block_Aw[(1 + ns):(ns * (N + 1))]
        for i in 1:N
            Jac2[:, :, i] = @view block_B[(1 + ns * (i - 1)):(ns * i), :]
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .+ block_Aw[(1 + ns * i):(ns * (i + 1))][bool_vec_s]
            Jac2[:, :, i] = @view block_B[(1 + (i - 1) * ns):(i * ns), :][bool_vec_s, :]
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

    ncon = (nc + num_real_bounds_s) * N
    nnzj = ncon * size(H, 2)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)

    J = LQJacobianOperator{T, M, AbstractArray{T}}(
        Jac1, Jac2, Jac3,
        N, nu, nc, num_real_bounds_s, 0,
        x1, x2, x3, y,
        SJ1, SJ2, SJ3, H_sub_block
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

function _build_implicit_dense_lq_dynamic_model(
    dnlp::LQDynamicData{T,V,M,MK}
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, MK <: AbstractMatrix{T}}
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

    dense_blocks = _build_block_matrices(A, B, K, N, w, nc)

    block_A   = dense_blocks.A
    block_B   = dense_blocks.B
    block_d   = dense_blocks.d
    block_Aw  = dense_blocks.Aw
    block_dw  = dense_blocks.dw
    block_KAw = dense_blocks.KAw

    H_blocks = _build_H_blocks(Q, R, block_A, block_B, block_Aw, S,Qf, K, s0, N)

    H  = H_blocks.H
    c0 = H_blocks.c0
    dense_blocks.h   .= H_blocks.block_h
    dense_blocks.h01 .= H_blocks.block_h01
    dense_blocks.h02 .= H_blocks.block_h02
    dense_blocks.h_constant  .= H_blocks.h_constant
    dense_blocks.h0_constant = H_blocks.h0_constant


    bool_vec_s        = (su .!= Inf .|| sl .!= -Inf)
    num_real_bounds_s   = sum(bool_vec_s)

    bool_vec_u       = (ul .!= -Inf .|| uu .!= Inf)
    num_real_bounds_u  = sum(bool_vec_u)

    G   = _init_similar(Q, nc * N, nu, T)
    Jac1 = _init_similar(Q, nc, nu, N, T)
    Jac2 = _init_similar(Q, num_real_bounds_s, nu, N, T)
    Jac3 = _init_similar(Q, num_real_bounds_u, nu, N, T)

    As0 = _init_similar(s0, ns * (N + 1), T)
    As0_bounds = _init_similar(s0, num_real_bounds_s * N, T)
    KAs0_bounds = _init_similar(s0, num_real_bounds_u * N, T)

    KBI        = _init_similar(Q, nu * N, nu, T)
    KAs0       = _init_similar(s0, nu * N, T)
    KAs0_block = _init_similar(s0, nu, T)
    KB         = _init_similar(Q, nu, nu, T)

    lcon = _init_similar(s0, (nc + num_real_bounds_s + num_real_bounds_u) * N, T)
    ucon = _init_similar(s0, (nc + num_real_bounds_s + num_real_bounds_u) * N, T)

    I_mat = _init_similar(Q, nu, nu, T)

    x1  = _init_similar(Q, nc, 1, N, T)
    x2  = _init_similar(Q, num_real_bounds_s, 1, N, T)
    x3  = _init_similar(Q, num_real_bounds_u, 1, N, T)
    y   = _init_similar(Q, nu, 1, N, T)

    SJ1   = _init_similar(Q, nc, nu, T)
    SJ2   = _init_similar(Q, num_real_bounds_s, nu, T)
    SJ3   = _init_similar(Q, num_real_bounds_u, nu, T)
    H_sub_block = _init_similar(Q, nu, nu, T)

    I_mat[LinearAlgebra.diagind(I_mat)] .= T(1)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    _set_G_blocks!(G, dl, du, block_B, block_A, block_d, block_Aw, block_dw, s0, E, F, K, N)

    for i in 1:N
        Jac1[:, :, i] = @view G[(1 + nc * (i - 1)):(nc * i), :]
    end

    LinearAlgebra.mul!(As0, block_A, s0)

    # Convert state variable constraints to algebraic constraints
    offset_s = nc * N
    if num_real_bounds_s == length(sl)
        As0_bounds .= As0[(1 + ns):ns * (N + 1)] .+ block_Aw[(1 + ns):(ns * (N + 1))]
        for i in 1:N
            Jac2[:, :, i] = @view block_B[(1 + ns * (i - 1)):(ns * i), :]
        end
    else
        for i in 1:N
            row_range = (1 + (i - 1) * num_real_bounds_s):(i * num_real_bounds_s)
            As0_bounds[row_range] .= As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .+ block_Aw[(1 + ns * i):(ns * (i + 1))][bool_vec_s]
            Jac2[:, :, i] = @view block_B[(1 + (i - 1) * ns):(i * ns), :][bool_vec_s, :]
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
        KAs0_bounds .= KAs0 .+ block_KAw
        for i in 1:N
            Jac3[:, :, i] = @view KBI[(1 + (i - 1) * nu):(i * nu), :]
        end
    else
        for i in 1:N
            row_range              = (1 + (i - 1) * num_real_bounds_u):(i * num_real_bounds_u)
            KAs0_bounds[row_range] = KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u] .+ block_KAw[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]
            Jac3[:, :, i] = @view KBI[(1 + (i - 1) * nu):(i * nu), :][bool_vec_u, :]
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
    ncon = (nc + num_real_bounds_s + num_real_bounds_u) * N
    nnzj = ncon * size(H, 1)
    nh   = size(H, 1)
    nnzh = div(nh * (nh + 1), 2)

    c = _init_similar(s0, nvar, T)
    c .= H_blocks.c

    J = LQJacobianOperator{T, M, AbstractArray{T}}(
        Jac1, Jac2, Jac3,
        N, nu, nc, num_real_bounds_s, num_real_bounds_u,
        x1, x2, x3, y,
        SJ1, SJ2, SJ3, H_sub_block
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
    A::M, B::M, K, N, w::V, nc
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}

    ns = size(A, 2)
    nu = size(B, 2)

    if K == nothing
        K = _init_similar(A, nu, ns, T)
    end

    # Define block matrices
    block_A   = _init_similar(A, ns * (N + 1), ns, T)
    block_B   = _init_similar(B, ns * N, nu, T)
    block_Aw  = _init_similar(w, ns * (N + 1), T)
    block_h   = _init_similar(A, nu * N, ns, T)
    block_h01 = _init_similar(A, ns, ns, T)
    block_h02 = _init_similar(w, ns, T)
    h_const   = _init_similar(w, nu * N, T)
    h0_const  = T(0)
    block_d   = _init_similar(A, nc * N, ns, T)
    block_dw  = _init_similar(w, nc * N, T)
    block_KA  = _init_similar(A, nu * N, ns, T)
    block_KAw = _init_similar(w, nu * N, T)

    A_k = copy(A)
    BK  = _init_similar(A, ns, ns, T)
    KA  = _init_similar(A, nu, ns, T)
    KAw = _init_similar(w, nu, T)
    Aw  = _init_similar(A, ns, T)

    AB_klast = _init_similar(A, size(B, 1), size(B, 2), T)
    AB_k     = _init_similar(A, size(B, 1), size(B, 2), T)

    block_B[1:ns, :] = B

    block_A[LinearAlgebra.diagind(block_A)] .= T(1)

    LinearAlgebra.mul!(BK, B, K)
    LinearAlgebra.axpy!(1, BK, A_k)

    A_klast  = copy(A_k)
    A_knext  = copy(A_k)

    block_A[(ns + 1):ns *2, :] = A_k
    LinearAlgebra.mul!(AB_k, A_k, B, 1, 0)

    block_B[(1 + ns):2 * ns, :] = AB_k
    AB_klast = copy(AB_k)
    # Fill the A and B matrices

    LinearAlgebra.mul!(KA, K, A_k)
    block_KA[1:nu, :] .= K
    block_KA[(1 + nu):(2 * nu), :] .= KA

    for i in 2:(N - 1)

        LinearAlgebra.mul!(AB_k, A_k, AB_klast)

        LinearAlgebra.mul!(A_knext, A_k, A_klast)

        block_A[(ns * i + 1):ns * (i + 1),:] = A_knext

        block_B[(1 + (i) * ns):((i + 1) * ns), :] = AB_k

        LinearAlgebra.mul!(KA, K, A_knext)
        block_KA[(1 + nu * i):(nu * (i + 1)), :] .= KA

        AB_klast = copy(AB_k)
        A_klast  = copy(A_knext)
    end

    LinearAlgebra.mul!(A_knext, A_k, A_klast)

    block_A[(ns * N + 1):ns * (N + 1), :] = A_knext

    for i in 1:N
        A_view = @view block_A[(1 + (i - 1) * ns):(i * ns), :]
        LinearAlgebra.mul!(Aw, A_view, w)
        for j in (i + 1):(N + 1)
            block_Aw[(1 + (j - 1) * ns):(j * ns)] .+= Aw
        end
    end

    for i in 1:N
        Aw_view = @view block_Aw[(1 + (i - 1) * ns):(i * ns)]
        LinearAlgebra.mul!(KAw, K, Aw_view)
        block_KAw[(1 + (i - 1) * nu):(i * nu)] .= KAw
    end

    DenseLQDynamicBlocks{T, V, M}(
        block_A,
        block_B,
        block_Aw,
        block_h,
        block_h01,
        block_h02,
        h_const,
        h0_const,
        block_d,
        block_dw,
        block_KA,
        block_KAw
    )
end

function _build_H_blocks(
    Q, R, block_A::M, block_B::M, Aw,
    S, Qf, K, s0, N
) where {T, M <: AbstractMatrix{T}}
    ns = size(Q, 1)
    nu = size(R, 1)

    if K == nothing
        K = _init_similar(Q, nu, ns, T)
    end

    H = _init_similar(block_A, nu * N, nu * N, T)

    # block_h01, block_h02, and block_h are stored in DenseLQDynamicBlocks to provide quick updates when redefining s0
    # block_h01 = A^T((Q + KTRK + 2 * SK))A where Q, K, R, S, and A are block matrices
    # block_h02 = A^T((Q + KTRK + 2 * SK))block_matrix_A w
    # block_h  = (QB + SKB + K^T R K B + K^T S^T B)^T A + (S + K^T R)^T A
    block_h01   = _init_similar(Q, ns, ns, T)
    block_h02   = _init_similar(s0, ns, T)
    block_h     = _init_similar(block_A, nu * N, ns, T)
    h_constant  = _init_similar(s0, nu * N, T)
    h0_constant = T(0)

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

    QB_block_vec = _init_similar(quad_term_B, ns * (N + 1), nu, T)
    h            = _init_similar(s0, nu * N, T)

    BTQA         = _init_similar(Q, nu, ns, T)
    RK_STA       = _init_similar(Q, nu, ns, T)
    BTQAw        = _init_similar(s0, nu, T)
    RK_STAw      = _init_similar(s0, nu, T)

    QA           = _init_similar(Q, ns, ns, T)
    KTRKA        = _init_similar(Q, ns, ns, T)
    SKA          = _init_similar(Q, ns, ns, T)
    KTSTA        = _init_similar(Q, ns, ns, T)

    QAw          = _init_similar(s0, ns, T)
    KTRKAw       = _init_similar(s0, ns, T)
    SKAw         = _init_similar(s0, ns, T)
    KTSTAw       = _init_similar(s0, ns, T)

    AQAs0        = _init_similar(s0, ns, T)

    LinearAlgebra.mul!(SK, S, K)
    LinearAlgebra.mul!(RK, R, K)
    LinearAlgebra.mul!(KTRK, K', RK)

    LinearAlgebra.axpy!(1.0, Q, quad_term)
    LinearAlgebra.axpy!(1.0, SK, quad_term)
    # axpy!(1.0, SK', quad_term) includes scalar operations because of the adjoint
    # .+= is more efficient with adjoint
    quad_term .+= SK'
    LinearAlgebra.axpy!(1.0, KTRK, quad_term)

    LinearAlgebra.copyto!(RK_ST, RK)
    RK_ST .+= S'

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
        # quad_term_B = QB + SKB + KTRKB + KTSTB = (Q + SK + KTRK + KTST) B
        fill!(QB_block_vec, T(0))
        rows_QB           = 1:(ns * (N - i))
        rows_QfB          = (1 + ns * (N - i)):(ns * (N - i + 1))

        QB_block_vec[(1 + ns * i):(ns * N), :]     = quad_term_B[rows_QB, :]
        QB_block_vec[(1 + ns * N):ns * (N + 1), :] = QfB[rows_QfB, :]

        LinearAlgebra.mul!(BTQA, QB_block_vec', block_A)
        LinearAlgebra.mul!(RK_STA, RK_ST, block_A[(ns * (i - 1) + 1):(ns * i), :])

        LinearAlgebra.mul!(BTQAw, QB_block_vec', Aw)
        LinearAlgebra.mul!(RK_STAw, RK_ST, Aw[(1 + ns * (i - 1)):(ns * i)])

        h_view = @view block_h[(1 + nu * (i - 1)):(nu * i), :]

        LinearAlgebra.axpy!(1, BTQA, h_view)
        LinearAlgebra.axpy!(1, RK_STA, h_view)

        h_constant_view = @view h_constant[(1 + nu * (i - 1)):(nu * i)]

        LinearAlgebra.axpy!(1, BTQAw, h_constant_view)
        LinearAlgebra.axpy!(1, RK_STAw, h_constant_view)

        A_view  = @view block_A[(1 + ns * (i - 1)):(ns * i), :]
        Aw_view = @view Aw[(1 + ns * (i - 1)):(ns * i)]

        LinearAlgebra.mul!(QA, Q, A_view)
        LinearAlgebra.mul!(KTRKA, KTRK, A_view)
        LinearAlgebra.mul!(SKA, SK, A_view)
        LinearAlgebra.mul!(KTSTA, SK', A_view)

        LinearAlgebra.mul!(QAw, Q, Aw_view)
        LinearAlgebra.mul!(KTRKAw, KTRK, Aw_view)
        LinearAlgebra.mul!(SKAw, SK, Aw_view)
        LinearAlgebra.mul!(KTSTAw, SK', Aw_view)

        LinearAlgebra.mul!(block_h01, A_view', QA, 1, 1)
        LinearAlgebra.mul!(block_h01, A_view', KTRKA, 1, 1)
        LinearAlgebra.mul!(block_h01, A_view', SKA, 1, 1)
        LinearAlgebra.mul!(block_h01, A_view', KTSTA, 1, 1)

        LinearAlgebra.mul!(block_h02, A_view', QAw, 1, 1)
        LinearAlgebra.mul!(block_h02, A_view', KTRKAw, 1, 1)
        LinearAlgebra.mul!(block_h02, A_view', SKAw, 1, 1)
        LinearAlgebra.mul!(block_h02, A_view', KTSTAw, 1, 1)

        h0_constant += LinearAlgebra.dot(Aw_view, QAw)
        h0_constant += LinearAlgebra.dot(Aw_view, KTRKAw)
        h0_constant += LinearAlgebra.dot(Aw_view, SKAw)
        h0_constant += LinearAlgebra.dot(Aw_view, KTSTAw)
    end

    A_view  = @view block_A[(1 + ns * N):(ns * (N + 1)), :]
    Aw_view = @view Aw[(1 + ns * N):(ns * (N + 1))]
    LinearAlgebra.mul!(QA, Qf, A_view)
    LinearAlgebra.mul!(block_h01, A_view', QA, 1, 1)

    LinearAlgebra.mul!(QAw, Qf, Aw_view)
    LinearAlgebra.mul!(block_h02, A_view', QAw, 1, 1)

    h0_constant += LinearAlgebra.dot(Aw_view, QAw)
    #LinearAlgebra.mul!(h0_constant, Aw_view', QAw, 1, 1)

    LinearAlgebra.mul!(h, block_h, s0)
    LinearAlgebra.mul!(AQAs0, block_h01, s0)
    h0 = LinearAlgebra.dot(AQAs0, s0)

    h0 += h0_constant
    h0 += LinearAlgebra.dot(block_h02, s0) * T(2)

    h += h_constant

    return (H = H, c = h, c0 = h0 / T(2), block_h = block_h, block_h01 = block_h01, block_h02 = block_h02, h_constant = h_constant, h0_constant = h0_constant / T(2))
end

function _set_G_blocks!(
    G, dl, du, block_B::M, block_A::M, block_d::M, block_Aw, block_dw,
    s0, E, F, K::MK, N
) where {T, M <: AbstractMatrix{T}, MK <: Nothing}
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    G[1:nc, :] = F

    EB   = _init_similar(block_B, nc, nu, T)
    EA   = _init_similar(block_B, nc, ns, T)
    d    = _init_similar(block_dw, nc, T)

    for i in 1:N
        if i != N
            B_row_range = (1 + (i - 1) * ns):(i * ns)
            B_sub_block = view(block_B, B_row_range, :)

            LinearAlgebra.mul!(EB, E, B_sub_block)
            G[(1 + nc * i):(nc * (i + 1)), :] = EB
        end
        A_view  = @view block_A[(1 + ns * (i - 1)):(ns * i), :]
        Aw_view = @view block_Aw[(1 + ns * (i - 1)):(ns * i)]
        LinearAlgebra.mul!(EA, E, A_view)
        LinearAlgebra.mul!(d, E, Aw_view)

        block_d[(1 + nc * (i - 1)):(nc * i), :] .= EA
        block_dw[(1 + nc * (i - 1)):(nc *i)]    .= d
    end
    LinearAlgebra.mul!(dl, block_d, s0, -1, 1)
    LinearAlgebra.mul!(du, block_d, s0, -1, 1)
    LinearAlgebra.axpy!(-1, block_dw, dl)
    LinearAlgebra.axpy!(-1, block_dw, du)
end

function _set_G_blocks!(
    G, dl, du, block_B, block_A, block_d, block_Aw, block_dw,
    s0, E, F, K::MK, N
) where {T, MK <: AbstractMatrix{T}}
    ns = size(E, 2)
    nu = size(F, 2)
    nc = size(E, 1)

    G[1:nc, :] = F

    E_FK  = _init_similar(E, nc, ns, T)
    E_FKA = _init_similar(E, nc, ns, T)
    FK    = _init_similar(E, nc, ns, T)
    EB    = _init_similar(E, nc, nu, T)
    d     = _init_similar(s0, nc, T)

    LinearAlgebra.copyto!(E_FK, E)
    LinearAlgebra.mul!(FK, F, K)
    LinearAlgebra.axpy!(1.0, FK, E_FK)

    for i in 1:N
        if i != N
            B_row_range = (1 + (i - 1) * ns):(i * ns)
            B_sub_block = view(block_B, B_row_range, :)

            LinearAlgebra.mul!(EB, E_FK, B_sub_block)
            G[(1 + nc * i):(nc * (i + 1)), :] = EB
        end
        A_view  = @view block_A[(1 + ns * (i - 1)):(ns * i), :]
        Aw_view = @view block_Aw[(1 + ns *(i - 1)):(ns * i)]
        LinearAlgebra.mul!(E_FKA, E_FK, A_view)
        LinearAlgebra.mul!(d, E_FK, Aw_view)

        block_d[(1 + nc * (i - 1)):(nc * i), :] .= E_FKA
        block_dw[(1 + nc * (i - 1)):(nc * i)]   .= d
    end
    LinearAlgebra.mul!(dl, block_d, s0, -1, 1)
    LinearAlgebra.mul!(du, block_d, s0, -1, 1)
    LinearAlgebra.axpy!(-1, block_dw, dl)
    LinearAlgebra.axpy!(-1, block_dw, du)
end

function _set_J1_dense!(J1, G, N)
    # Only used for explicit Jacobian, not implicit Jacobian
    nu = size(G, 2)
    nc = Int(size(G, 1) / N)

    for i in 1:N
        col_range = (1 + nu * (i - 1)):(nu * i)
        J1[(1 + nc * (i - 1)):nc * N, col_range] = G[1:((N - i + 1) * nc),:]
    end

end
