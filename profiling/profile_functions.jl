using Revise
using DynamicNLPModels, LinearAlgebra, Random, SparseArrays, Profile, PProf

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



lqdm_d = build_lqdm(1000, 10, 50; implicit=false)
lqdm_imp = build_lqdm(1000, 10, 50; implicit=true)
x = rand(size(lqdm_d.data.A, 1))
y = rand(size(lqdm_d.data.A, 2))
y_imp = copy(y)
x_imp = copy(x)
J_imp = get_jacobian(lqdm_imp)

@profile mul!(x_imp, J_imp, y_imp)
pprof()
