function test_mul(lq_dense, lq_dense_imp)
    dnlp = lq_dense.dynamic_data
    N    = dnlp.N
    nu   = dnlp.nu

    J     = get_jacobian(lq_dense)
    J_imp = get_jacobian(lq_dense_imp)

    Random.seed!(10)
    x = rand(nu * N)
    y = rand(size(J, 1))

    x_imp = similar(lq_dense_imp.dynamic_data.s0, length(x))
    y_imp = similar(lq_dense_imp.dynamic_data.s0, length(y))
    LinearAlgebra.copyto!(x_imp, x)
    LinearAlgebra.copyto!(y_imp, y)

    LinearAlgebra.mul!(y, J, x)
    LinearAlgebra.mul!(y_imp, J_imp, x_imp)

    @test y ≈ Vector(y_imp) atol = 1e-14

    x = rand(nu * N)
    y = rand(size(J, 1))

    x_imp = similar(lq_dense_imp.dynamic_data.s0, length(x))
    y_imp = similar(lq_dense_imp.dynamic_data.s0, length(y))
    LinearAlgebra.copyto!(x_imp, x)
    LinearAlgebra.copyto!(y_imp, y)

    LinearAlgebra.mul!(x, J', y)
    LinearAlgebra.mul!(x_imp, J_imp', y_imp)

    @test x ≈ Vector(x_imp) atol = 1e-14
end

function test_add_jtsj(lq_dense, lq_dense_imp)
    dnlp = lq_dense.dynamic_data
    N    = dnlp.N
    nu   = dnlp.nu

    H     = zeros(nu * N, nu * N)

    Random.seed!(10)
    J     = get_jacobian(lq_dense)
    J_imp = get_jacobian(lq_dense_imp)
    ΣJ    = similar(J); fill!(ΣJ, 0)

    x     = rand(size(J, 1))

    H_imp = similar(lq_dense_imp.data.H, nu * N, nu * N); fill!(H_imp, 0)
    x_imp = similar(lq_dense_imp.dynamic_data.s0, length(x));
    LinearAlgebra.copyto!(x_imp, x)

    LinearAlgebra.mul!(ΣJ, Diagonal(x), J)
    LinearAlgebra.mul!(H, J', ΣJ)

    add_jtsj!(H_imp, J_imp, x_imp)

    @test LowerTriangular(Array(H_imp)) ≈ LowerTriangular(H) atol = 1e-10
end

function dynamic_data_to_CUDA(dnlp::LQDynamicData)
    s0c = CuVector{Float64}(undef, length(dnlp.s0))
    Ac  = CuArray{Float64}(undef, size(dnlp.A))
    Bc  = CuArray{Float64}(undef, size(dnlp.B))
    Qc  = CuArray{Float64}(undef, size(dnlp.Q))
    Rc  = CuArray{Float64}(undef, size(dnlp.R))
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

    if dnlp.K != nothing
        Kc  = CuArray{Float64}(undef, size(dnlp.K))
        LinearAlgebra.copyto!(Kc, dnlp.K)
    else
        Kc = nothing
    end

    LQDynamicData(s0c, Ac, Bc, Qc, Rc, dnlp.N; Qf = Qfc, S = Sc,
    E = Ec, F = Fc, K = Kc, sl = slc, su = suc, ul = ulc, uu = uuc, gl = glc, gu = guc
    )
end

function test_sparse_support(lqdm)
    d = lqdm.dynamic_data

    (lqdm_sparse_data = SparseLQDynamicModel(d.s0, sparse(d.A), sparse(d.B), sparse(d.Q), sparse(d.R), d.N;
        sl = d.sl, ul = d.ul, su = d.su, uu = d.uu, Qf = sparse(d.Qf), K = (d.K == nothing ? nothing : sparse(d.K)),
        S = sparse(d.S), E = sparse(d.E), F = sparse(d.F), gl = d.gl, gu = d.gu))

    @test lqdm.data.H ≈ lqdm_sparse_data.data.H atol = 1e-10
    @test lqdm.data.A ≈ lqdm_sparse_data.data.A atol = 1e-10
end

function test_dense_reset_s0(dnlp, lq_dense, new_s0)
    lq_dense_test   = DenseLQDynamicModel(dnlp)
    dnlp.s0 .= new_s0
    lq_dense_new_s0 = DenseLQDynamicModel(dnlp)

    reset_s0!(lq_dense_test, new_s0)

    @test lq_dense_test.data.H ≈ lq_dense_new_s0.data.H atol = 1e-10
    @test lq_dense_test.data.A ≈ lq_dense_new_s0.data.A atol = 1e-10
    @test lq_dense_test.data.c ≈ lq_dense_new_s0.data.c atol = 1e-10
    @test lq_dense_test.data.c0 ≈ lq_dense_new_s0.data.c0 atol = 1e-10

    @test lq_dense_test.meta.lcon ≈ lq_dense_new_s0.meta.lcon atol = 1e-8
    @test lq_dense_test.meta.ucon ≈ lq_dense_new_s0.meta.ucon atol = 1e-8
    @test lq_dense_test.dynamic_data.s0 == lq_dense_new_s0.dynamic_data.s0
end

function test_sparse_reset_s0(dnlp, lq_sparse, new_s0)
    reset_s0!(lq_sparse, new_s0)

    @test lq_sparse.dynamic_data.s0 == new_s0
end

function runtests(model, dnlp, lq_sparse, lq_dense, lq_sparse_from_data, lq_dense_from_data, N, ns, nu)
    optimize!(model)
    solution_ref_sparse           = madnlp(lq_sparse, max_iter=100)
    solution_ref_dense            = madnlp(lq_dense, max_iter=100)
    solution_ref_sparse_from_data = madnlp(lq_sparse_from_data, max_iter=100)
    solution_ref_dense_from_data  = madnlp(lq_dense_from_data, max_iter=100)

    @test objective_value(model) ≈ solution_ref_sparse.objective atol = 1e-7
    @test objective_value(model) ≈ solution_ref_dense.objective atol = 1e-5
    @test objective_value(model) ≈ solution_ref_sparse_from_data.objective atol = 1e-7
    @test objective_value(model) ≈ solution_ref_dense_from_data.objective atol = 1e-5

    @test solution_ref_sparse.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense.solution atol =  1e-6
    @test solution_ref_sparse_from_data.solution[(ns * (N + 1) + 1):(ns * (N + 1) + nu*N)] ≈ solution_ref_dense_from_data.solution atol =  1e-6

    # Test get_u and get_s functions with no K matrix
    s_values = value.(all_variables(model)[1:(ns * (N + 1))])
    u_values = value.(all_variables(model)[(1 + ns * (N + 1)):(ns * (N + 1) + nu * N)])

    @test s_values ≈ get_s(solution_ref_sparse, lq_sparse) atol = 1e-7
    @test u_values ≈ get_u(solution_ref_sparse, lq_sparse) atol = 1e-7
    @test s_values ≈ get_s(solution_ref_dense, lq_dense) atol = 1e-5
    @test u_values ≈ get_u(solution_ref_dense, lq_dense) atol = 1e-5

    test_sparse_support(lq_sparse)

    lq_dense_imp = DenseLQDynamicModel(dnlp; implicit = true)

    imp_test_set = []
    push!(imp_test_set, lq_dense_imp)

    if CUDA.has_cuda_gpu()
        dnlp_cuda     = dynamic_data_to_CUDA(dnlp)
        lq_dense_cuda = DenseLQDynamicModel(dnlp_cuda; implicit=true)
        push!(imp_test_set, lq_dense_cuda)
    end

    @testset "Test mul and add_jtsj!" for lq_imp in imp_test_set
        test_mul(lq_dense, lq_imp)
        test_add_jtsj(lq_dense, lq_imp)
    end

    new_s0 = copy(dnlp.s0) .+ .5
    test_dense_reset_s0(dnlp, lq_dense, new_s0)

    new_s0 = copy(dnlp.s0) .+ 1
    test_sparse_reset_s0(dnlp, lq_sparse, new_s0)
end
