using KernelAbstractions, CUDA, CUDAKernels

function add_JTSJ!(J, S, SJ, H)
    SJ = S .* J

    LinearAlgebra.mul!(H, J', SJ)
end

@kernel function JTSJ_kernel!(J, S, H)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(J))
    for k in 1:size(J, 1)
        tmp_sum += J[k, i] * S[k] * J[k, j]
    end

    H[i,j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function add_JTSJ!(J, S, H)
    if size(J, 1) != length(S)
        println("Matrix size mismatch!")
        return nothing
    end
    device = KernelAbstractions.get_device(J)

    kernel! = JTSJ_kernel!(device)
    #n = size(H, 1)
    ev= kernel!(J, S, H, ndrange=size(H))
    wait(ev)
end

function time_GPU_kernel(dim1, dim2)
    J1 = rand(dim1, dim2)
    S1 = rand(dim1)
    H1 = zeros(dim2, dim2)
    H2  = copy(H1)

    SJ1 = zeros(dim1, dim2)

    Jcu = CuArray(J1)
    Scu = CuArray(S1)
    Hcu1 = CuArray(H1)

    Hcu2 = CuArray(H1)
    SJcu  = CuArray(SJ1)

    add_JTSJ!(Jcu, Scu, Hcu1)
    a = @elapsed add_JTSJ!(Jcu, Scu, Hcu1)
    add_JTSJ!(Jcu, Scu, SJcu, Hcu2)
    b = @elapsed add_JTSJ!(Jcu, Scu, SJcu, Hcu2)

    return a, b
end

nc_range = [100, 1000, 5000, 10000, 100000]
nu       = 50

kernel_vals50 = []
mul_vals50    = []
for i in nc_range
    a, b = time_GPU_kernel(i, nu)
    push!(kernel_vals50, a)
    push!(mul_vals50, b)
end


nc_range = [100, 1000, 5000, 10000, 100000]
nu       = 200

kernel_vals200 = []
mul_vals200    = []
for i in nc_range
    a, b = time_GPU_kernel(i, nu)
    push!(kernel_vals200, a)
    push!(mul_vals200, b)
end

nc_range = [100, 1000, 5000, 10000, 100000]
nu       = 500

kernel_vals500 = []
mul_vals500    = []
for i in nc_range
    a, b = time_GPU_kernel(i, nu)
    push!(kernel_vals500, a)
    push!(mul_vals500, b)
end



using Plots, LaTeXStrings
plot(nc_range, kernel_vals50, yaxis=:log, xaxis=:log, label="custom Kernel", legend=:topleft)
plot!(nc_range, mul_vals50, label="gemm")
title!(L"$J^T \Sigma J$ for size(J, 2) =  50")
xlabel!("size(J, 1)")
ylabel!("Time (s)")
savefig("kernel50.png")

using Plots, LaTeXStrings
plot(nc_range, kernel_vals200, yaxis=:log, xaxis=:log, label="custom Kernel", legend=:topleft)
plot!(nc_range, mul_vals200, label="gemm")
title!(L"$J^T \Sigma J$ for size(J, 2) =  200")
xlabel!("size(J, 1)")
ylabel!("Time (s)")
savefig("kernel200.png")

using Plots, LaTeXStrings
plot(nc_range, kernel_vals500, yaxis=:log, xaxis=:log, label="custom Kernel", legend=:topleft)
plot!(nc_range, mul_vals500, label="gemm")
title!(L"$J^T \Sigma J$ for size(J, 2) =  500")
xlabel!("size(J, 1)")
ylabel!("Time (s)")
savefig("kernel500.png")
