"""
get_u(solution_ref, lqdm::SparseLQDynamicModel) -> u <: vector
get_u(solution_ref, lqdm::DenseLQDynamicModel) -> u <: vector

Query the solution `u` from the solver. If `K = nothing`, the solution for `u` is queried from `solution_ref.solution`

If `K <: AbstractMatrix`, `solution_ref.solution` returns `v`, and `get_u` solves for `u` using the `K` matrix (and the `A` and `B` matrices if `lqdm <: DenseLQDynamicModel`)
"""
function get_u(
    solver_status,
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    MK <: AbstractMatrix{T},
}

    solution = solver_status.solution
    ns = lqdm.dynamic_data.ns
    nu = lqdm.dynamic_data.nu
    N = lqdm.dynamic_data.N
    K = lqdm.dynamic_data.K

    u = zeros(T, nu * N)

    for i = 1:N
        start_v = (i - 1) * nu + 1
        end_v = i * nu
        start_s = (i - 1) * ns + 1
        end_s = i * ns

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
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    M4 <: AbstractMatrix{T},
    MK <: AbstractMatrix{T},
}

    dnlp = lqdm.dynamic_data

    N = dnlp.N
    ns = dnlp.ns
    nu = dnlp.nu
    K = dnlp.K

    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B
    block_Aw = lqdm.blocks.Aw

    v = solver_status.solution

    As0 = zeros(T, ns * (N + 1))
    Bv = zeros(T, ns)
    s = zeros(T, ns * (N + 1))

    for i = 1:N
        B_row_range = (1 + (i - 1) * ns):(i * ns)
        B_sub_block = view(block_B, B_row_range, :)

        for j = 1:(N - i + 1)
            v_sub_vec = v[(1 + nu * (j - 1)):(nu * j)]
            LinearAlgebra.mul!(Bv, B_sub_block, v_sub_vec)

            s[(1 + ns * (i + j - 1)):(ns * (i + j))] .+= Bv
        end
    end

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.axpy!(1, As0, s)
    LinearAlgebra.axpy!(1, block_Aw, s)

    Ks = _init_similar(dnlp.s0, size(K, 1), T)
    u = copy(v)
    for i = 1:N
        LinearAlgebra.mul!(Ks, K, s[(1 + ns * (i - 1)):(ns * i)])
        u[(1 + nu * (i - 1)):(nu * i)] .+= Ks
    end

    return u
end

function get_u(
    solver_status,
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    MK <: Nothing,
}

    solution = solver_status.solution
    ns = lqdm.dynamic_data.ns
    nu = lqdm.dynamic_data.nu
    N = lqdm.dynamic_data.N

    u = solution[(ns * (N + 1) + 1):end]
    return u
end

function get_u(
    solver_status,
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    M4 <: AbstractMatrix{T},
    MK <: Nothing,
}
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
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    MK <: Union{Nothing, AbstractMatrix},
}

    solution = solver_status.solution
    ns = lqdm.dynamic_data.ns
    N = lqdm.dynamic_data.N

    s = solution[1:(ns * (N + 1))]
    return s
end

function get_s(
    solver_status,
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
) where {
    T,
    V <: AbstractVector{T},
    M1 <: AbstractMatrix{T},
    M2 <: AbstractMatrix{T},
    M3 <: AbstractMatrix{T},
    M4 <: AbstractMatrix{T},
    MK <: Union{Nothing, AbstractMatrix},
}

    dnlp = lqdm.dynamic_data

    N = dnlp.N
    ns = dnlp.ns
    nu = dnlp.nu

    block_A = lqdm.blocks.A
    block_B = lqdm.blocks.B
    block_Aw = lqdm.blocks.Aw

    v = solver_status.solution

    As0 = zeros(T, ns * (N + 1))
    Bv = zeros(T, ns)
    s = zeros(T, ns * (N + 1))

    for i = 1:N
        B_row_range = (1 + (i - 1) * ns):(i * ns)
        B_sub_block = view(block_B, B_row_range, :)

        for j = 1:(N - i + 1)
            v_sub_vec = v[(1 + nu * (j - 1)):(nu * j)]
            LinearAlgebra.mul!(Bv, B_sub_block, v_sub_vec)

            s[(1 + ns * (i + j - 1)):(ns * (i + j))] .+= Bv
        end
    end

    LinearAlgebra.mul!(As0, block_A, dnlp.s0)
    LinearAlgebra.axpy!(1, As0, s)
    LinearAlgebra.axpy!(1, block_Aw, s)

    return s
end

for field in fieldnames(LQDynamicData)
    method = Symbol("get_", field)
    @eval begin
        @doc """
            $($method)(LQDynamicData)
            $($method)(SparseLQDynamicModel)
            $($method)(DenseLQDynamicModel)
        Return the value of $($(QuoteNode(field))) from `LQDynamicData` or `SparseLQDynamicModel.dynamic_data` or `DenseLQDynamicModel.dynamic_data`
        """
        $method(dyn_data::LQDynamicData) = getproperty(dyn_data, $(QuoteNode(field)))
    end
    @eval $method(dyn_model::SparseLQDynamicModel) = $method(dyn_model.dynamic_data)
    @eval $method(dyn_model::DenseLQDynamicModel) = $method(dyn_model.dynamic_data)
    @eval export $method
end

for field in [:A, :B, :Q, :R, :Qf, :E, :F, :S, :K]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, row, col, val)
            $($method)(SparseLQDynamicModel, row, col, val)
            $($method)(DenseLQDynamicModel, row, col, val)
        Set the value of entry $($(QuoteNode(field)))[row, col] to val for `LQDynamicData`, `SparseLQDynamicModel.dynamic_data`, or `DenseLQDynamicModel.dynamic_data`
        """
        $method(dyn_data::LQDynamicData, row, col, val) = (dyn_data.$field[row, col] = val)
    end
    @eval $method(dyn_model::SparseLQDynamicModel, row, col, val) =
        (dyn_model.dynamic_data.$field[row, col] = val)
    @eval $method(dyn_model::DenseLQDynamicModel, row, col, val) =
        (dyn_model.dynamic_data.$field[row, col] = val)
    @eval export $method
end

for field in [:s0, :sl, :su, :ul, :uu, :gl, :gu]
    method = Symbol("set_", field, "!")
    @eval begin
        @doc """
            $($method)(LQDynamicData, index, val)
            $($method)(SparseLQDynamicModel, index, val)
            $($method)(DenseLQDynamicModel, index, val)
        Set the value of entry $($(QuoteNode(field)))[index] to val for `LQDynamicData`, `SparseLQDynamicModel.dynamic_data`, or `DenseLQDynamicModel.dynamic_data`
        """
        $method(dyn_data::LQDynamicData, index, val) = (dyn_data.$field[index] = val)
    end
    @eval $method(dyn_model::SparseLQDynamicModel, index, val) =
        (dyn_model.dynamic_data.$field[index] = val)
    @eval $method(dyn_model::DenseLQDynamicModel, index, val) =
        (dyn_model.dynamic_data.$field[index] = val)
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
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
    fill_structure!(qp.data.H, rows, cols)
    return rows, cols
end


function NLPModels.hess_structure!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, V, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
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
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
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
) where {T, V, M1 <: SparseMatrixCSC, M2 <: SparseMatrixCSC, M3 <: AbstractMatrix}
    fill_structure!(qp.data.A, rows, cols)
    return rows, cols
end

function NLPModels.jac_structure!(
    qp::DenseLQDynamicModel{T, V, M1, M2, M3},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, V, M1 <: Matrix, M2 <: Matrix, M3 <: Matrix}
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

function _dnlp_unsafe_wrap(
    tensor::A,
    dims::Tuple,
    shift = 1,
) where {T, A <: AbstractArray{T}}
    return unsafe_wrap(Matrix{T}, pointer(tensor, shift), dims)
end

function _dnlp_unsafe_wrap(
    tensor::A,
    dims::Tuple,
    shift = 1,
) where {T, A <: CUDA.CuArray{T, 3, CUDA.Mem.DeviceBuffer}}
    return unsafe_wrap(
        CUDA.CuArray{T, 2, CUDA.Mem.DeviceBuffer},
        pointer(tensor, shift),
        dims,
    )
end

function LinearAlgebra.mul!(
    y::V,
    Jac::LQJacobianOperator{T, M, A},
    x::V,
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    fill!(y, zero(T))

    J1 = Jac.truncated_jac1
    J2 = Jac.truncated_jac2
    J3 = Jac.truncated_jac3

    N = Jac.N
    nu = Jac.nu
    nc = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    for i = 1:N
        sub_B1 = _dnlp_unsafe_wrap(J1, (nc, nu), (1 + (i - 1) * (nc * nu)))
        sub_B2 = _dnlp_unsafe_wrap(J2, (nsc, nu), (1 + (i - 1) * (nsc * nu)))
        sub_B3 = _dnlp_unsafe_wrap(J3, (nuc, nu), (1 + (i - 1) * (nuc * nu)))

        for j = 1:(N - i + 1)
            sub_x = view(x, (1 + (j - 1) * nu):(j * nu))
            LinearAlgebra.mul!(
                view(y, (1 + nc * (j + i - 2)):(nc * (j + i - 1))),
                sub_B1,
                sub_x,
                1,
                1,
            )
            LinearAlgebra.mul!(
                view(y, (1 + nc * N + nsc * (j + i - 2)):(nc * N + nsc * (j + i - 1))),
                sub_B2,
                sub_x,
                1,
                1,
            )
            LinearAlgebra.mul!(
                view(
                    y,
                    (1 + nc * N + nsc * N + nuc * (j + i - 2)):(nc * N + nsc * N + nuc * (j + i - 1)),
                ),
                sub_B3,
                sub_x,
                1,
                1,
            )
        end
    end
end

function LinearAlgebra.mul!(
    x::V,
    Jac::LQJacobianOperator{T, M, A},
    y::V,
) where {
    T,
    V <: CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer},
    M <: AbstractMatrix{T},
    A <: AbstractArray{T},
}

    J1 = Jac.truncated_jac1
    J2 = Jac.truncated_jac2
    J3 = Jac.truncated_jac3

    N = Jac.N
    nu = Jac.nu
    nc = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    x1 = Jac.x1
    x2 = Jac.x2
    x3 = Jac.x3
    y1 = Jac.y

    fill!(x1, zero(T))
    fill!(x2, zero(T))
    fill!(x3, zero(T))

    for i = 1:N
        y1 .= y[(1 + (i - 1) * nu):(i * nu)]

        x1_view = view(x1, :, :, i:N)
        x2_view = view(x2, :, :, i:N)
        x3_view = view(x3, :, :, i:N)

        J1_view = view(J1, :, :, 1:(N - i + 1))
        J2_view = view(J2, :, :, 1:(N - i + 1))
        J3_view = view(J3, :, :, 1:(N - i + 1))

        y1_view = view(y1, :, :, i:N)

        CUBLAS.gemm_strided_batched!('N', 'N', 1, J1_view, y1_view, 1, x1_view)
        CUBLAS.gemm_strided_batched!('N', 'N', 1, J2_view, y1_view, 1, x2_view)
        CUBLAS.gemm_strided_batched!('N', 'N', 1, J3_view, y1_view, 1, x3_view)
    end

    x[1:(nc * N)] .= reshape(x1, nc * N)
    x[(1 + nc * N):((nc + nsc) * N)] .= reshape(x2, nsc * N)
    x[(1 + (nc + nsc) * N):((nc + nsc + nuc) * N)] .= reshape(x3, nuc * N)
end

function LinearAlgebra.mul!(
    y::V,
    Jac::LinearOperators.AdjointLinearOperator{T, LQJacobianOperator{T, M, A}},
    x::V,
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    fill!(y, zero(T))

    jac_op = get_jacobian(Jac)

    J1 = jac_op.truncated_jac1
    J2 = jac_op.truncated_jac2
    J3 = jac_op.truncated_jac3

    N = jac_op.N
    nu = jac_op.nu
    nc = jac_op.nc
    nsc = jac_op.nsc
    nuc = jac_op.nuc

    for i = 1:N
        sub_B1 = _dnlp_unsafe_wrap(J1, (nc, nu), (1 + (i - 1) * (nc * nu)))
        sub_B2 = _dnlp_unsafe_wrap(J2, (nsc, nu), (1 + (i - 1) * (nsc * nu)))
        sub_B3 = _dnlp_unsafe_wrap(J3, (nuc, nu), (1 + (i - 1) * (nuc * nu)))

        for j = 1:(N - i + 1)

            x1 = view(x, (1 + (j + i - 2) * nc):((j + i - 1) * nc))
            x2 = view(x, (1 + nc * N + (j + i - 2) * nsc):(nc * N + (j + i - 1) * nsc))
            x3 = view(
                x,
                (1 + nc * N + nsc * N + (j + i - 2) * nuc):(nc * N + nsc * N + (j + i - 1) * nuc),
            )

            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j)), sub_B1', x1, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j)), sub_B2', x2, 1, 1)
            LinearAlgebra.mul!(view(y, (1 + nu * (j - 1)):(nu * j)), sub_B3', x3, 1, 1)
        end
    end
end


function LinearAlgebra.mul!(
    y::V,
    Jac::LinearOperators.AdjointLinearOperator{T, LQJacobianOperator{T, M, A}},
    x::V,
) where {
    T,
    V <: CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer},
    M <: AbstractMatrix{T},
    A <: AbstractArray{T},
}
    fill!(y, zero(T))

    jac_op = get_jacobian(Jac)

    J1 = jac_op.truncated_jac1
    J2 = jac_op.truncated_jac2
    J3 = jac_op.truncated_jac3

    N = jac_op.N
    nu = jac_op.nu
    nc = jac_op.nc
    nsc = jac_op.nsc
    nuc = jac_op.nuc

    x1 = jac_op.x1
    x2 = jac_op.x2
    x3 = jac_op.x3
    y1 = jac_op.y


    x1 .= reshape(x[1:(nc * N)], (nc, 1, N))
    x2 .= reshape(x[(1 + nc * N):((nc + nsc) * N)], (nsc, 1, N))
    x3 .= reshape(x[(1 + (nc + nsc) * N):((nc + nsc + nuc) * N)], (nuc, 1, N))

    for i = 1:N
        fill!(y1, zero(T))

        y1_view = view(y1, :, :, 1:(N - i + 1))

        x1_view = view(x1, :, :, i:N)
        x2_view = view(x2, :, :, i:N)
        x3_view = view(x3, :, :, i:N)

        J1_view = view(J1, :, :, 1:(N - i + 1))
        J2_view = view(J2, :, :, 1:(N - i + 1))
        J3_view = view(J3, :, :, 1:(N - i + 1))

        CUBLAS.gemm_strided_batched!('T', 'N', 1, J1_view, x1_view, 1, y1_view)
        CUBLAS.gemm_strided_batched!('T', 'N', 1, J2_view, x2_view, 1, y1_view)
        CUBLAS.gemm_strided_batched!('T', 'N', 1, J3_view, x3_view, 1, y1_view)

        view(y, (1 + (i - 1) * nu):(i * nu)) .= sum(y1_view, dims = (2, 3))
    end

end

"""
    get_jacobian(lqdm::DenseLQDynamicModel) -> LQJacobianOperator
    get_jacobian(Jac::AdjointLinearOpeartor{T, LQJacobianOperator}) -> LQJacobianOperator

Gets the `LQJacobianOperator` from `DenseLQDynamicModel` (if the `QPdata` contains a `LQJacobian Operator`)
or returns the `LQJacobian Operator` from the adjoint of the `LQJacobianOperator`
"""
function get_jacobian(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
) where {T, V, M1, M2, M3, M4, MK}
    return lqdm.data.A
end

function get_jacobian(
    Jac::LinearOperators.AdjointLinearOperator{T, LQJacobianOperator{T, M, A}},
) where {T, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    return Jac'
end

function Base.length(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    return length(Jac.truncated_jac1) +
           length(Jac.truncated_jac2) +
           length(Jac.truncated_jac3)
end

function Base.size(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    return (
        size(Jac.truncated_jac1, 1) +
        size(Jac.truncated_jac2, 1) +
        size(Jac.truncated_jac3, 1),
        size(Jac.truncated_jac1, 2),
    )
end

function Base.eltype(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractMatrix{T}}
    return T
end

function Base.isreal(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractMatrix{T}}
    return isreal(Jac.truncated_jac1) &&
           isreal(Jac.truncated_jac2) &&
           isreal(Jac.truncated_jac3)
end

function Base.show(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractMatrix{T}}
    show(Jac.truncated_jac1)
end

function Base.display(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractMatrix{T}}
    display(Jac.truncated_jac1)
end
"""
    LinearOperators.reset!(Jac::LQJacobianOperator{T, V, M})

Resets the values of attributes `SJ1`, `SJ2`, and `SJ3` to zero
"""
function LinearOperators.reset!(
    Jac::LQJacobianOperator{T, M, A},
) where {T, M <: AbstractMatrix{T}, A <: AbstractMatrix{T}}
    fill!(Jac.SJ1, T(0))
    fill!(Jac.SJ2, T(0))
    fill!(Jac.SJ3, T(0))
end

function NLPModels.jac_op(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
    x::V,
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
    Jac::LQJacobianOperator{T, M, A},
    Σ::V,
    alpha::Number = 1,
    beta::Number = 1,
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, A <: AbstractArray{T}}

    J1 = Jac.truncated_jac1
    J2 = Jac.truncated_jac2
    J3 = Jac.truncated_jac3

    N = Jac.N
    nu = Jac.nu
    nc = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    ΣJ1 = Jac.SJ1
    ΣJ2 = Jac.SJ2
    ΣJ3 = Jac.SJ3

    LinearAlgebra.lmul!(beta, H)

    for i = 1:N
        left_block1 = _dnlp_unsafe_wrap(J1, (nc, nu), (1 + (i - 1) * (nc * nu)))
        left_block2 = _dnlp_unsafe_wrap(J2, (nsc, nu), (1 + (i - 1) * (nsc * nu)))
        left_block3 = _dnlp_unsafe_wrap(J3, (nuc, nu), (1 + (i - 1) * (nuc * nu)))

        for j = 1:(N + 1 - i)
            Σ_range1 = (1 + (N - j) * nc):((N - j + 1) * nc)
            Σ_range2 = (1 + nc * N + (N - j) * nsc):(nc * N + (N - j + 1) * nsc)
            Σ_range3 =
                (1 + (nc + nsc) * N + (N - j) * nuc):((nc + nsc) * N + (N - j + 1) * nuc)

            ΣJ1 .= left_block1 .* view(Σ, Σ_range1)
            ΣJ2 .= left_block2 .* view(Σ, Σ_range2)
            ΣJ3 .= left_block3 .* view(Σ, Σ_range3)

            for k = 1:(N - j - i + 2)
                right_block1 =
                    _dnlp_unsafe_wrap(J1, (nc, nu), (1 + (k + i - 2) * (nc * nu)))
                right_block2 =
                    _dnlp_unsafe_wrap(J2, (nsc, nu), (1 + (k + i - 2) * (nsc * nu)))
                right_block3 =
                    _dnlp_unsafe_wrap(J3, (nuc, nu), (1 + (k + i - 2) * (nuc * nu)))

                row_range = (1 + nu * (N - i - j + 1)):(nu * (N - i - j + 2))
                col_range = (1 + nu * (N - i - k - j + 2)):(nu * (N - i - k - j + 3))

                LinearAlgebra.mul!(
                    view(H, row_range, col_range),
                    ΣJ1',
                    right_block1,
                    alpha,
                    1,
                )
                LinearAlgebra.mul!(
                    view(H, row_range, col_range),
                    ΣJ2',
                    right_block2,
                    alpha,
                    1,
                )
                LinearAlgebra.mul!(
                    view(H, row_range, col_range),
                    ΣJ3',
                    right_block3,
                    alpha,
                    1,
                )
            end
        end
    end
end

function add_jtsj!(
    H::M,
    Jac::LQJacobianOperator{T, M, A},
    Σ::V,
    alpha::Number = 1,
    beta::Number = 1,
) where {T, V <: CUDA.CuVector, M <: AbstractMatrix{T}, A <: AbstractArray{T}}

    J1 = Jac.truncated_jac1
    J2 = Jac.truncated_jac2
    J3 = Jac.truncated_jac3

    N = Jac.N
    nu = Jac.nu
    nc = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    ΣJ1 = Jac.SJ1
    ΣJ2 = Jac.SJ2
    ΣJ3 = Jac.SJ3

    H_sub_block = Jac.H_sub_block

    LinearAlgebra.lmul!(beta, H)

    for i = 1:N
        left_block1 = view(J1, :, :, i)
        left_block2 = view(J2, :, :, i)
        left_block3 = view(J3, :, :, i)

        for j = 1:(N + 1 - i)
            Σ_range1 = (1 + (N - j) * nc):((N - j + 1) * nc)
            Σ_range2 = (1 + nc * N + (N - j) * nsc):(nc * N + (N - j + 1) * nsc)
            Σ_range3 =
                (1 + (nc + nsc) * N + (N - j) * nuc):((nc + nsc) * N + (N - j + 1) * nuc)

            ΣJ1 .= left_block1 .* view(Σ, Σ_range1)
            ΣJ2 .= left_block2 .* view(Σ, Σ_range2)
            ΣJ3 .= left_block3 .* view(Σ, Σ_range3)

            for k = 1:(N - j - i + 2)
                right_block1 = view(J1, :, :, (k + i - 1))
                right_block2 = view(J2, :, :, (k + i - 1))
                right_block3 = view(J3, :, :, (k + i - 1))

                row_range = (1 + nu * (N - i - j + 1)):(nu * (N - i - j + 2))
                col_range = (1 + nu * (N - i - k - j + 2)):(nu * (N - i - k - j + 3))

                LinearAlgebra.mul!(H_sub_block, ΣJ1', right_block1)
                H[row_range, col_range] .+= alpha .* H_sub_block

                LinearAlgebra.mul!(H_sub_block, ΣJ2', right_block2)
                H[row_range, col_range] .+= alpha .* H_sub_block

                LinearAlgebra.mul!(H_sub_block, ΣJ3', right_block3)
                H[row_range, col_range] .+= alpha .* H_sub_block
            end
        end
    end
end

"""
    reset_s0!(lqdm::SparseLQDynamicModel, s0)
    reset_s0!(lqdm::DenseLQDynamicModel, s0)

Resets `s0` within `lqdm.dynamic_data`. For a `SparseLQDynamicModel`, this updates the variable bounds which fix the value of `s0`.
For a `DenseLQDynamicModel`, also resets the constraint bounds on the Jacobian and resets the linear and constant terms within the
objective function (i.e., `lqdm.data.c` and `lqdm.data.c0`). This provides a way to update the model after each sample period.
"""
function reset_s0!(
    lqdm::SparseLQDynamicModel{T, V, M1, M2, M3, MK},
    s0::V,
) where {T, V <: AbstractVector{T}, M1, M2, M3, MK}
    dnlp = lqdm.dynamic_data
    ns = dnlp.ns

    lqdm.dynamic_data.s0 .= s0

    lqdm.meta.lvar[1:ns] .= s0
    lqdm.meta.uvar[1:ns] .= s0
end

function reset_s0!(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
    s0::V,
) where {T, V <: AbstractVector{T}, M1, M2, M3, M4, MK <: Nothing}

    dnlp = lqdm.dynamic_data
    dense_blocks = lqdm.blocks

    N = dnlp.N
    ns = dnlp.ns
    nu = dnlp.nu
    E = dnlp.E
    F = dnlp.E
    ul = dnlp.ul
    uu = dnlp.uu
    sl = dnlp.sl
    su = dnlp.su
    gl = dnlp.gl
    gu = dnlp.gu
    nc = size(E, 1)

    # Get matrices for multiplying by s0
    block_A = dense_blocks.A
    block_Aw = dense_blocks.Aw
    block_h = dense_blocks.h
    block_h0 = dense_blocks.h01
    block_d = dense_blocks.d
    block_dw = dense_blocks.dw
    block_h02 = dense_blocks.h02
    h_constant = dense_blocks.h_constant
    h0_constant = dense_blocks.h0_constant

    lcon = lqdm.meta.lcon
    ucon = lqdm.meta.ucon

    # Reset s0
    lqdm.dynamic_data.s0 .= s0

    As0 = _init_similar(s0, ns * (N + 1), T)
    Qs0 = _init_similar(s0, ns, T)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    bool_vec_s = (sl .!= -Inf .|| su .!= Inf)
    nsc = sum(bool_vec_s)

    sl = sl[bool_vec_s]
    su = su[bool_vec_s]

    LinearAlgebra.mul!(dl, block_d, s0, -1, 1)
    LinearAlgebra.mul!(du, block_d, s0, -1, 1)

    # Reset constraint bounds corresponding to E and F matrices
    lcon[1:(nc * N)] .= dl
    ucon[1:(nc * N)] .= du

    lcon[1:(nc * N)] .-= block_dw
    ucon[1:(nc * N)] .-= block_dw

    LinearAlgebra.mul!(As0, block_A, s0)

    # reset linear term
    LinearAlgebra.mul!(lqdm.data.c, block_h, s0)
    lqdm.data.c += h_constant

    # reset constant term
    LinearAlgebra.mul!(Qs0, block_h0, s0)
    lqdm.data.c0 = LinearAlgebra.dot(s0, Qs0) / T(2)

    lqdm.data.c0 += h0_constant
    lqdm.data.c0 += LinearAlgebra.dot(s0, block_h02)

    for i = 1:N
        # Reset bounds on constraints from state variable bounds
        lcon[(1 + nc * N + nsc * (i - 1)):(nc * N + nsc * i)] .=
            sl .- As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .-
            block_Aw[(1 + ns * i):((i + 1) * ns)][bool_vec_s]
        ucon[(1 + nc * N + nsc * (i - 1)):(nc * N + nsc * i)] .=
            su .- As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .-
            block_Aw[(1 + ns * i):((i + 1) * ns)][bool_vec_s]
    end
end

function reset_s0!(
    lqdm::DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK},
    s0::V,
) where {T, V <: AbstractVector{T}, M1, M2, M3, M4, MK <: AbstractMatrix{T}}
    dnlp = lqdm.dynamic_data
    dense_blocks = lqdm.blocks

    N = dnlp.N
    ns = dnlp.ns
    nu = dnlp.nu
    E = dnlp.E
    F = dnlp.E
    K = dnlp.K
    ul = dnlp.ul
    uu = dnlp.uu
    sl = dnlp.sl
    su = dnlp.su
    gl = dnlp.gl
    gu = dnlp.gu
    nc = size(E, 1)

    # Get matrices for multiplying by s0
    block_A = dense_blocks.A
    block_Aw = dense_blocks.Aw
    block_h = dense_blocks.h
    block_h0 = dense_blocks.h01
    block_d = dense_blocks.d
    block_dw = dense_blocks.dw
    block_KA = dense_blocks.KA
    block_KAw = dense_blocks.KAw
    block_h02 = dense_blocks.h02
    h_constant = dense_blocks.h_constant
    h0_constant = dense_blocks.h0_constant

    lcon = lqdm.meta.lcon
    ucon = lqdm.meta.ucon

    # Reset s0
    lqdm.dynamic_data.s0 .= s0

    lqdm.data.c0 += LinearAlgebra.dot(s0, block_h02)
    As0 = _init_similar(s0, ns * (N + 1), T)
    Qs0 = _init_similar(s0, ns, T)
    KAs0 = _init_similar(s0, nu * N, T)

    dl = repeat(gl, N)
    du = repeat(gu, N)

    bool_vec_s = (sl .!= -Inf .|| su .!= Inf)
    nsc = sum(bool_vec_s)

    bool_vec_u = (ul .!= -Inf .|| uu .!= Inf)
    nuc = sum(bool_vec_u)

    sl = sl[bool_vec_s]
    su = su[bool_vec_s]

    ul = ul[bool_vec_u]
    uu = uu[bool_vec_u]

    LinearAlgebra.mul!(dl, block_d, s0, -1, 1)
    LinearAlgebra.mul!(du, block_d, s0, -1, 1)

    # Reset constraint bounds corresponding to E and F matrices
    lcon[1:(nc * N)] .= dl
    ucon[1:(nc * N)] .= du

    lcon[1:(nc * N)] .-= block_dw
    ucon[1:(nc * N)] .-= block_dw

    LinearAlgebra.mul!(As0, block_A, s0)
    LinearAlgebra.mul!(KAs0, block_KA, s0)

    # reset linear term
    LinearAlgebra.mul!(lqdm.data.c, block_h, s0)
    lqdm.data.c += h_constant

    # reset constant term
    LinearAlgebra.mul!(Qs0, block_h0, s0)
    lqdm.data.c0 = LinearAlgebra.dot(s0, Qs0) / T(2)

    lqdm.data.c0 += h0_constant
    lqdm.data.c0 += LinearAlgebra.dot(s0, block_h02)

    for i = 1:N
        # Reset bounds on constraints from state variable bounds
        lcon[(1 + nc * N + nsc * (i - 1)):(nc * N + nsc * i)] .=
            sl .- As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .-
            block_Aw[(1 + i * ns):((i + 1) * ns)][bool_vec_s]
        ucon[(1 + nc * N + nsc * (i - 1)):(nc * N + nsc * i)] .=
            su .- As0[(1 + ns * i):(ns * (i + 1))][bool_vec_s] .-
            block_Aw[(1 + i * ns):((i + 1) * ns)][bool_vec_s]

        # Reset bounds on constraints from input variable bounds
        lcon[(1 + (nc + nsc) * N + nuc * (i - 1)):((nc + nsc) * N + nuc * i)] .=
            ul .- KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u] .-
            block_KAw[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]
        ucon[(1 + (nc + nsc) * N + nuc * (i - 1)):((nc + nsc) * N + nuc * i)] .=
            uu .- KAs0[(1 + nu * (i - 1)):(nu * i)][bool_vec_u] .-
            block_KAw[(1 + nu * (i - 1)):(nu * i)][bool_vec_u]
    end
end
