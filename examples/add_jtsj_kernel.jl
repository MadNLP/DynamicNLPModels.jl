using Revise
using KernelAbstractions, CUDA, CUDAKernels
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays
using Test

include("build_thinplate.jl")

@kernel function JTSJ_kernel!(J1, J2, J3, S, H, i, N, nu, nc, nsc, nuc)
    j, k, l, m = @index(Global, NTuple)

    # j = 1:i, but all of our calls will treat it as i:-1:1
    # this can be obtained by taking (j - i) * -1 + 1
    # k = 1:(N - i + 1), but all of our calls will treat is as i:N
    # this can be obtained by taking k + (i - 1)
    # l = 1:nu (rows of Hessian block)
    # m = 1:nu (cols of Hessian block)

    tmp_sum1 = zero(eltype(J1))
    tmp_sum2 = zero(eltype(J1))
    tmp_sum3 = zero(eltype(J1))

    @inbounds begin
    for o in 1:nc
        tmp_sum1 += J1[o, l, (j - i) * -1 + 1] * S[o + ((k + i - 1) - 1) * nc] * J1[o, m, i]
    end

    for o in 1:nsc
        tmp_sum2 += J2[o, l, (j - i) * -1 + 1] * S[o + ((k + i - 1) - 1) * nsc + nc * N] * J2[o, m, i]
    end

    for o in 1:nuc
        tmp_sum3 += J3[o, l, (j - i) * -1 + 1] * S[o + ((k + i - 1) - 1) * nuc + (nc + nsc) * N] * J3[o, m, i]
    end

    # j_ind = (j - i) * -1 + 1
    # k_ind = k - i + 1
    # H_row_val = l + (k + j - 2) * nu
    # H_col_val = m + (k - 1) * nu

    H[(l + (k + j - 2) * nu), (m + (k - 1) * nu)] += tmp_sum1 + tmp_sum2 + tmp_sum3
    end
end

# Creating a wrapper kernel for launching with error checks
function add_JTSJ!(J1, J2, J3, S, H, i, N, nu, nc, nsc, nuc)

    device = KernelAbstractions.get_device(J1)

    kernel! = JTSJ_kernel!(device)

    # Not sure how to define ndrange here...
    ev = kernel!(J1, J2, J3, S, H, i, N, nu, nc, nsc, nuc, ndrange=(i, (N - i + 1), nu, nu))
    wait(ev)
end

function add_JTSJ!_kernel(J, S, H)

    J1  = J.truncated_jac1
    J2  = J.truncated_jac2
    J3  = J.truncated_jac3
    N   = J.N
    nu  = J.nu
    nc  = J.nc
    nsc = J.nsc
    nuc = J.nuc

    CUDA.@time for i in 1:div(N, 2)
        add_JTSJ!(J1, J2, J3, S, H, i, N, nu, nc, nsc, nuc)
    end
    CUDA.@time for i in div(N, 2):N
        add_JTSJ!(J1, J2, J3, S, H, i, N, nu, nc, nsc, nuc)
    end
end


N = 50
nu = 10
ns = 100

@time lqdm_d   = build_lqdm(ns, nu, N; implicit=false)
println("built full Jacobian lqdm")
@time lqdm_imp = build_lqdm(ns, nu, N; implicit=true)
println("built implicit Jacobian lqdm")

Random.seed!(10)

H     = zeros(nu * N, nu * N)
H_imp = zeros(nu * N, nu * N)


J     = get_jacobian(lqdm_d)
J_imp = get_jacobian(lqdm_imp)
ΣJ    = similar(J); fill!(ΣJ, 0)

x     = rand(size(J, 1))


H     = zeros(nu * N, nu * N)
H_imp = zeros(nu * N, nu * N)
@time add_JTSJ!_kernel(J_imp, x, H_imp)
@time jtsj_mul!(H, J, x, ΣJ)



#@test LowerTriangular(H) ≈ LowerTriangular(H_imp) atol = 1e-15
