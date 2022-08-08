using KernelAbstractions, CUDA, CUDAKernels

@kernel function JTSJ_kernel!(J, S, H)
    i, j, k, l, m = @index(Global, NTuple)

    # i = i:i
    # j = i:-1:1
    # k = i:N
    # l = 1:nu (rows of Hessian block)
    # m = 1:nu (cols of Hessian block)

    J1  = J.truncated_jac1
    J2  = J.truncated_jac2
    J3  = J.truncated_jac3
    N   = J.N
    nu  = J.nu
    nc  = J.nc
    nsc = J.nsc
    nuc = J.nuc

    tmp_sum1 = zero(eltype(J1))
    tmp_sum2 = zero(eltype(J1))
    tmp_sum3 = zero(eltype(J1))

    for o in 1:nc
        tmp_sum1 += J1[o, l, i] * S[o + (k - 1) * nc] * J1[o, m, j]
    end

    for o in 1:nsc
        tmp_sum2 += J2[o, l, i] * S[o + (k - 1) * nsc + nc * N] * J2[o, m, j]
    end

    for o in 1:nuc
        tmp_sum3 += J3[o, l, i] * S[o + (k - 1) * nuc + (nc + nsc) * N] * J3[o, m, j]
    end

    # j_ind = (j - i) * -1 + 1
    # k_ind = k - i + 1
    # H_row_val = l + (k_ind + j_ind - 2) * nu = l + (k - i + (j - i) * -1) * nu = l + (k - j) * nu
    # H_col_val = m + (k_ind - 1) * nu = m + (k - i) * nu

    H[(l + (k - j) * nu), (m + (k - i) * nu)] += tmp_sum1 + tmp_sum2 + tmp_sum3

end


# Creating a wrapper kernel for launching with error checks
function add_JTSJ!(J, S, H)
    if size(J, 1) != length(S)
        println("Matrix size mismatch!")
        return nothing
    end
    device = KernelAbstractions.get_device(J.truncated_jac1)

    kernel! = JTSJ_kernel!(device)

    # Not sure how to define ndrange here...
    ev = kernel!(J, S, H, ndrange=???)
    wait(ev)
end

# ORIGINAL FUNCTION:
#=
function add_jtsj!(
    H::M,
    Jac::LQJacobianOperator{T, M, A},
    Σ::AbstractVector{T},
    alpha::Number = 1,
    beta::Number = 1
) where {T, M <: AbstractMatrix{T}, A <: AbstractArray{T}}
    J1  = Jac.truncated_jac1
    J2  = Jac.truncated_jac2
    J3  = Jac.truncated_jac3
    N   = Jac.N
    nu  = Jac.nu
    nc  = Jac.nc
    nsc = Jac.nsc
    nuc = Jac.nuc

    ΣJ1 = Jac.SJ1
    ΣJ2 = Jac.SJ2
    ΣJ3 = Jac.SJ3

    if beta != 1
        for i in 1:(nu * N)
            for j in 1:(nu * N)
                H[i, j] = H[i, j] * beta
            end
        end
    end

    for i in 1:N
        right_block1 = J1[:, :, i]
        right_block2 = J2[:, :, i]
        right_block3 = J3[:, :, i]
        #println()
        #println("Iteration $i")
        if i <= cld(N, 2)
            for (j_ind, j) in enumerate(i:-1:1)
                left_block1 = J1[:, :, j]
                left_block2 = J2[:, :, j]
                left_block3 = J3[:, :, j]

                for (k_ind, k) in enumerate(i:N)
                    row_range = (1 + (k_ind + j_ind - 2) * nu):((k_ind + j_ind - 1) * nu)
                    col_range = (1 + (k_ind - 1) * nu):(k_ind * nu)

                    #println("leftblock = ", j, " sigma = ", k, " rightblock = ", i, " row = ", (k_ind + j_ind - 1), " col = ", k_ind)

                    for (mJ, mH) in enumerate(col_range)
                        for (lJ, lH) in enumerate(row_range)
                            for o in 1:nc
                                H[lH, mH] += left_block1[o, lJ] * right_block1[o, mJ] * alpha * Σ[o + (k - 1) * nc]
                            end

                            for o in 1:nsc
                                H[lH, mH] += left_block2[o, lJ] * right_block2[o, mJ] * alpha * Σ[o + (k - 1) * nsc + nc * N]
                            end

                            for o in 1:nuc
                                H[lH, mH] += left_block3[o, lJ] * right_block3[o, mJ] * alpha * Σ[o + (k - 1) * nuc + (nsc + nc) * N]
                            end
                        end
                    end
                end
                #println()
            end
        else
            for (j_ind, j) in enumerate(i:N)
                for (k_ind, k) in enumerate(i:-1:1)
                    left_block1 = J1[:, :, k]
                    left_block2 = J2[:, :, k]
                    left_block3 = J3[:, :, k]

                    row_range = (1 + (k_ind + j_ind - 2) * nu):((k_ind + j_ind -1) * nu)
                    col_range = (1 + (j_ind - 1) * nu):(j_ind * nu)
                    #println("leftblock = ", k, " sigma = ", j, " rightblock = ", i, " row = ", k_ind + j_ind - 1, " col = ", j_ind)

                    for (mJ, mH) in enumerate(col_range)
                        for (lJ, lH) in enumerate(row_range)
                            for o in 1:nc
                                H[lH, mH] += left_block1[o, lJ] * right_block1[o, mJ] * alpha * Σ[o + (j - 1) * nc]
                            end

                            for o in 1:nsc
                                H[lH, mH] += left_block2[o, lJ] * right_block2[o, mJ] * alpha * Σ[o + (j - 1) * nsc + nc * N]
                            end

                            for o in 1:nuc
                                H[lH, mH] += left_block3[o, lJ] * right_block3[o, mJ] * alpha * Σ[o + (j - 1) * nuc + (nsc + nc) * N]
                            end
                        end
                    end
                end
                #println()
            end
        end
    end
end
=#
