using Revise
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays, CUDA
using Test

function jtsj_mul!(H, J, x, SJ)
    SJ .= x .* J

    LinearAlgebra.mul!(H, J', SJ, 1, 1)
end

include("build_thinplate.jl")

function build_lqdm(ns, nu, N; implicit=false)

    d = zeros(ns, N+1)
    dfunc = (x,y)->100*sin(2*pi*(4*x/N-12*y/ns)) + 400
    for l in 1:(N+1)
        for m in 1:ns
            d[m,l] = dfunc(m,l)
        end
    end

    K = 0.5 * Matrix(I, nu, ns)
    S = -.001 * Matrix(I, ns, nu)
    E = zeros(ns - 1, ns)

    for i in 1:(ns - 1)
        E[i, i] = 1
        E[i, i + 1] = -1
    end

    F = zeros(ns - 1, nu)

    gl_val = -100. / (ns / nu)
    gu_val = 100. / (ns / nu)

    gl = fill(gl_val, ns - 1)
    gu = fill(gu_val, ns - 1)

    gl .*= rand(.8:.00001:1, ns - 1)
    gu .*= rand(.8:.00001:1, ns - 1)

    if implicit
        lqdm = build_thinplate(ns, nu, N, .001, .01; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu, implicit = implicit)
    else
        lqdm = build_thinplate(ns, nu, N, .001, .01; d = d, Tbar = 400., dense = true, sl = 300., su = 500., ul = -140., uu = 140., K = K, S = S, E = E, F = F, gl = gl, gu = gu)
    end

    return lqdm
end


function dynamic_data_to_CUDA(dnlp::LQDynamicData)
    s0c = CuVector{Float64}(undef, length(dnlp.s0))
    Ac  = CuArray{Float64}(undef, size(dnlp.A))
    Bc  = CuArray{Float64}(undef, size(dnlp.B))
    Qc  = CuArray{Float64}(undef, size(dnlp.Q))
    Rc  = CuArray{Float64}(undef, size(dnlp.R))
    Kc  = CuArray{Float64}(undef, size(dnlp.K))
    Sc  = CuArray{Float64}(undef, size(dnlp.S))
    Ec  = CuArray{Float64}(undef, size(dnlp.E))
    Fc  = CuArray{Float64}(undef, size(dnlp.F))
    Qfc = CuArray{Float64}(undef, size(dnlp.Qf))
    glc = CuVector{Float64}(undef, length(dnlp.gl))
    guc = CuVector{Float64}(undef, length(dnlp.gu))
    ulc = CuVector{Float64}(undef, length(dnlp.ul))
    uuc = CuVector{Float64}(undef, length(dnlp.uu))
    slc = CuVector{Float64}(undef, length(dnlp.sl))
    suc = CuVector{Float64}(undef, length(dnlp.su))

    LinearAlgebra.copyto!(Ac, dnlp.A)
    LinearAlgebra.copyto!(Bc, dnlp.B)
    LinearAlgebra.copyto!(Qc, dnlp.Q)
    LinearAlgebra.copyto!(Rc, dnlp.R)
    LinearAlgebra.copyto!(s0c, dnlp.s0)
    LinearAlgebra.copyto!(Kc, dnlp.K)
    LinearAlgebra.copyto!(Sc, dnlp.S)
    LinearAlgebra.copyto!(Ec, dnlp.E)
    LinearAlgebra.copyto!(Fc, dnlp.F)
    LinearAlgebra.copyto!(Qfc, dnlp.Qf)
    LinearAlgebra.copyto!(glc, dnlp.gl)
    LinearAlgebra.copyto!(guc, dnlp.gu)
    LinearAlgebra.copyto!(ulc, dnlp.ul)
    LinearAlgebra.copyto!(uuc, dnlp.uu)
    LinearAlgebra.copyto!(slc, dnlp.sl)
    LinearAlgebra.copyto!(suc, dnlp.su)


    LQDynamicData(s0c, Ac, Bc, Qc, Rc, dnlp.N; Qf = Qfc, S = Sc,
    E = Ec, F = Fc, K = Kc, sl = slc, su = suc, ul = ulc, uu = uuc, gl = glc, gu = guc
    )
end
N = 5
nu = 3
ns = 3

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

jtsj_mul!(H, J, x, ΣJ)
add_jtsj!(H_imp, J_imp, x, 1, 1)

@test LowerTriangular(H) ≈ LowerTriangular(H_imp) atol=1e-14
