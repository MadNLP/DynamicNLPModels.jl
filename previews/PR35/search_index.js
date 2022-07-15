var documenterSearchIndex = {"docs":
[{"location":"api/#API-Manual","page":"API Manual","title":"API Manual","text":"","category":"section"},{"location":"api/","page":"API Manual","title":"API Manual","text":"Modules = [DynamicNLPModels]","category":"page"},{"location":"api/#DynamicNLPModels.DenseLQDynamicBlocks","page":"API Manual","title":"DynamicNLPModels.DenseLQDynamicBlocks","text":"Struct containing block A and B matrices used in creating the DenseLQDynamicModel. These matrices are given by Jerez, Kerrigan, and Constantinides in section 4 of \"A sparse and condensed QP formulation for predictive control of LTI systems\" (doi:10.1016/j.automatica.2012.03.010).\n\nA is a ns(N+1) x ns matrix and B is a ns(N) x nu matrix containing the first column of the B block matrix in the above text. Note that the first block of zeros is omitted.\n\n\n\n\n\n","category":"type"},{"location":"api/#DynamicNLPModels.DenseLQDynamicModel-Union{Tuple{LQDynamicData{T, V, M}}, Tuple{MK}, Tuple{M}, Tuple{V}, Tuple{T}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, MK<:Union{Nothing, AbstractMatrix{T}}}","page":"API Manual","title":"DynamicNLPModels.DenseLQDynamicModel","text":"DenseLQDynamicModel(dnlp::LQDynamicData; implicit = false)    -> DenseLQDynamicModel\nDenseLQDynamicModel(s0, A, B, Q, R, N; implicit = false ...) -> DenseLQDynamicModel\n\nA constructor for building a DenseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel\n\nInput data is for the problem of the form\n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                u_i = Kx_i + v_i  forall i = 0 1  N - 1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\nData is converted to the form\n\n    minimize    frac12 u^T H u + h^T u + h0\n    subject to  Jz le g\n                ul le u le uu\n\nResulting H, J, h, and h0 matrices are stored within QuadraticModels.QPData as H, A, c, and c0 attributes respectively\n\nIf K is defined, then u variables are replaced by v variables. The bounds on u are transformed into algebraic constraints, and u can be queried by get_u and get_s within DynamicNLPModels.jl\n\nKeyword argument implicit = false determines how the Jacobian is stored within the QPData. If implicit = false, the full, dense Jacobian matrix is stored. If implicit = true, only the first nu columns of the Jacobian are stored with the Linear Operator LQJacobianOperator.\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.LQDynamicData","page":"API Manual","title":"DynamicNLPModels.LQDynamicData","text":"LQDynamicData{T,V,M,MK} <: AbstractLQDynData{T,V}\n\nA struct to represent the features of the optimization problem\n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                u_i = Kx_i + v_i  forall i = 0 1  N - 1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\nAttributes include:\n\ns0: initial state of system\nA : constraint matrix for system states\nB : constraint matrix for system inputs\nQ : objective function matrix for system states from 1:(N-1)\nR : objective function matrix for system inputs from 1:(N-1)\nN : number of time steps\nQf: objective function matrix for system state at time N\nS : objective function matrix for system states and inputs\nns: number of state variables\nnu: number of input varaibles\nE : constraint matrix for state variables\nF : constraint matrix for input variables\nK : feedback gain matrix\nsl: vector of lower bounds on state variables\nsu: vector of upper bounds on state variables\nul: vector of lower bounds on input variables\nuu: vector of upper bounds on input variables\ngl: vector of lower bounds on constraints\ngu: vector of upper bounds on constraints\n\nsee also LQDynamicData(s0, A, B, Q, R, N; ...)\n\n\n\n\n\n","category":"type"},{"location":"api/#DynamicNLPModels.LQDynamicData-Union{Tuple{MK}, Tuple{M}, Tuple{V}, Tuple{T}, Tuple{V, M, M, M, M, Any}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, MK<:Union{Nothing, AbstractMatrix{T}}}","page":"API Manual","title":"DynamicNLPModels.LQDynamicData","text":"LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, V, M, MK}\n\nA constructor for building an object of type LQDynamicData for the optimization problem\n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  forall i=0 1  N - 1\n                u_i = Kx_i + v_i  forall i = 0 1  N - 1\n                gl le E s_i + F u_i le gu forall i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\ns0: initial state of system\nA : constraint matrix for system states\nB : constraint matrix for system inputs\nQ : objective function matrix for system states from 1:(N-1)\nR : objective function matrix for system inputs from 1:(N-1)\nN : number of time steps\n\nThe following attributes of the LQDynamicData type are detected automatically from the length of s0 and size of R\n\nns: number of state variables\nnu: number of input varaibles\n\nThe following keyward arguments are also accepted\n\nQf = Q: objective function matrix for system state at time N; dimensions must be ns x ns\nS  = nothing: objective function matrix for system state and inputs\nE  = zeros(eltype(Q), 0, ns)  : constraint matrix for state variables\nF  = zeros(eltype(Q), 0, nu)  : constraint matrix for input variables\nK  = nothing       : feedback gain matrix\nsl = fill(-Inf, ns): vector of lower bounds on state variables\nsu = fill(Inf, ns) : vector of upper bounds on state variables\nul = fill(-Inf, nu): vector of lower bounds on input variables\nuu = fill(Inf, nu) : vector of upper bounds on input variables\ngl = fill(-Inf, size(E, 1)) : vector of lower bounds on constraints\ngu = fill(Inf, size(E, 1))  : vector of upper bounds on constraints\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.LQJacobianOperator","page":"API Manual","title":"DynamicNLPModels.LQJacobianOperator","text":"LQJacobianOperator{T, V, M}\n\nStruct for storing the implicit Jacobian matrix. All data for the Jacobian can be stored in the first nu columns of J. This struct contains the needed data and storage arrays for calculating Jx, J^T x, and J^T Sigma J. Jx and J^T x are performed through extensions to LinearAlgebra.mul!().\n\n\n\nAttributes\n\nJac: Matrix of first nu columns of the Jacobian\nN  : number of time steps\nnu : number of inputs\nnc : number of algebraic constraints of the form gl <= Es + Fu <= gu\nnsc: number of bounded state variables\nnuc: number of bounded input variables (if K is defined)\nSJ1: placeholder for storing data when calculating ΣJ\nSJ2: placeholder for storing data when calculating ΣJ\nSJ3: placeholder for storing data when calculating ΣJ\n\n\n\n\n\n","category":"type"},{"location":"api/#DynamicNLPModels.SparseLQDynamicModel-Union{Tuple{LQDynamicData{T, V, M}}, Tuple{MK}, Tuple{M}, Tuple{V}, Tuple{T}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, MK<:Union{Nothing, AbstractMatrix{T}}}","page":"API Manual","title":"DynamicNLPModels.SparseLQDynamicModel","text":"SparseLQDynamicModel(dnlp::LQDynamicData)    -> SparseLQDynamicModel\nSparseLQDynamicModel(s0, A, B, Q, R, N; ...) -> SparseLQDynamicModel\n\nA constructor for building a SparseLQDynamicModel <: QuadraticModels.AbstractQuadraticModel Input data is for the problem of the form\n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                u_i = Kx_i + v_i  forall i = 0 1  N - 1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\nData is converted to the form\n\n    minimize    frac12 z^T H z\n    subject to  lcon le Jz le ucon\n                lvar le z le uvar\n\nResulting H and J matrices are stored as QuadraticModels.QPData within the SparseLQDynamicModel struct and variable and constraint limits are stored within NLPModels.NLPModelMeta\n\nIf K is defined, then u variables are replaced by v variables, and u can be queried by get_u and get_s within DynamicNLPModels.jl\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels._set_sparse_H!-Union{Tuple{M}, Tuple{T}, Tuple{Any, Any, Any, M, M, Any}} where {T, M<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels._set_sparse_H!","text":"_set_sparse_H!(H_colptr, H_rowval, H_nzval, Q, R, N; Qf = Q, S = zeros(T, size(Q, 1), size(R, 1))\n\nset the data needed to build a SparseArrays.SparseMatrixCSC matrix. Hcolptr, Hrowval, and Hnzval are set so that they can be passed to SparseMatrixCSC() to obtain the H matrix such that  z^T H z = sum{i=1}^{N-1} si^T Q s + sum{i=1}^{N-1} u^T R u + sN^T Qf sn .\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels._set_sparse_J!-Union{Tuple{MK}, Tuple{T}, Tuple{Any, Any, Any, Any, Any, Any, Any, MK, Any, Any, Any}} where {T, MK<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels._set_sparse_J!","text":"_set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, bool_vec, N, nb)\n_set_sparse_J!(J_colptr, J_rowval, J_nzval, A, B, E, F, K, N)\n\nset the data needed to build a SparseArrays.SparseMatrixCSC matrix. Jcolptr, Jrowval, and J_nzval are set so that they can be passed to SparseMatrixCSC() to obtain the Jacobian, J. The Jacobian contains the data for the following constraints:\n\nAsi + Bui = s{i + 1} gl <= Esi + Fui <= getu\n\nIf K is defined, then this matrix also contains the constraints ul <= Kxi + vi <= uu\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.add_jtsj!-Union{Tuple{M}, Tuple{V}, Tuple{T}, Tuple{M, DynamicNLPModels.LQJacobianOperator{T, V, M}, V}, Tuple{M, DynamicNLPModels.LQJacobianOperator{T, V, M}, V, Number}, Tuple{M, DynamicNLPModels.LQJacobianOperator{T, V, M}, V, Number, Number}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels.add_jtsj!","text":"add_jtsj!(H::M, Jac::LQJacobianOperator{T, V, M}, Σ::V)\n\nGenerates Jac' Σ Jac and adds it to the matrix H.\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_A-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_A","text":"get_A(LQDynamicData)\nget_A(SparseLQDynamicModel)\nget_A(DenseLQDynamicModel)\n\nReturn the value A from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_B-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_B","text":"get_B(LQDynamicData)\nget_B(SparseLQDynamicModel)\nget_B(DenseLQDynamicModel)\n\nReturn the value B from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_E-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_E","text":"get_E(LQDynamicData)\nget_E(SparseLQDynamicModel)\nget_E(DenseLQDynamicModel)\n\nReturn the value E from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_F-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_F","text":"get_F(LQDynamicData)\nget_F(SparseLQDynamicModel)\nget_F(DenseLQDynamicModel)\n\nReturn the value F from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_K-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_K","text":"get_K(LQDynamicData)\nget_K(SparseLQDynamicModel)\nget_K(DenseLQDynamicModel)\n\nReturn the value K from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_N-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_N","text":"get_N(LQDynamicData)\nget_N(SparseLQDynamicModel)\nget_N(DenseLQDynamicModel)\n\nReturn the value N from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_Q-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_Q","text":"get_Q(LQDynamicData)\nget_Q(SparseLQDynamicModel)\nget_Q(DenseLQDynamicModel)\n\nReturn the value Q from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_Qf-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_Qf","text":"get_Qf(LQDynamicData)\nget_Qf(SparseLQDynamicModel)\nget_Qf(DenseLQDynamicModel)\n\nReturn the value Qf from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_R-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_R","text":"get_R(LQDynamicData)\nget_R(SparseLQDynamicModel)\nget_R(DenseLQDynamicModel)\n\nReturn the value R from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_S-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_S","text":"get_S(LQDynamicData)\nget_S(SparseLQDynamicModel)\nget_S(DenseLQDynamicModel)\n\nReturn the value S from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_gl-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_gl","text":"get_gl(LQDynamicData)\nget_gl(SparseLQDynamicModel)\nget_gl(DenseLQDynamicModel)\n\nReturn the value gl from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_gu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_gu","text":"get_gu(LQDynamicData)\nget_gu(SparseLQDynamicModel)\nget_gu(DenseLQDynamicModel)\n\nReturn the value gu from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_jacobian-Union{Tuple{DenseLQDynamicModel{T, V, M1, M2, M3, M4, MK}}, Tuple{MK}, Tuple{M4}, Tuple{M3}, Tuple{M2}, Tuple{M1}, Tuple{V}, Tuple{T}} where {T, V, M1, M2, M3, M4, MK}","page":"API Manual","title":"DynamicNLPModels.get_jacobian","text":"get_jacobian(lqdm::DenseLQDynamicModel) -> LQJacobianOperator\nget_jacobian(Jac::AdjointLinearOpeartor{T, LQJacobianOperator}) -> LQJacobianOperator\n\nGets the LQJacobianOperator from DenseLQDynamicModel (if the QPdata contains a LQJacobian Operator) or returns the LQJacobian Operator from the adjoint of the LQJacobianOperator\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_ns-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_ns","text":"get_ns(LQDynamicData)\nget_ns(SparseLQDynamicModel)\nget_ns(DenseLQDynamicModel)\n\nReturn the value ns from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_nu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_nu","text":"get_nu(LQDynamicData)\nget_nu(SparseLQDynamicModel)\nget_nu(DenseLQDynamicModel)\n\nReturn the value nu from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_s-Union{Tuple{MK}, Tuple{M3}, Tuple{M2}, Tuple{M1}, Tuple{V}, Tuple{T}, Tuple{Any, SparseLQDynamicModel{T, V, M1, M2, M3, MK}}} where {T, V<:AbstractVector{T}, M1<:AbstractMatrix{T}, M2<:AbstractMatrix{T}, M3<:AbstractMatrix{T}, MK<:Union{Nothing, AbstractMatrix}}","page":"API Manual","title":"DynamicNLPModels.get_s","text":"get_s(solution_ref, lqdm::SparseLQDynamicModel) -> s <: vector\nget_s(solution_ref, lqdm::DenseLQDynamicModel) -> s <: vector\n\nQuery the solution s from the solver. If lqdm <: SparseLQDynamicModel, the solution is queried directly from solution_ref.solution If lqdm <: DenseLQDynamicModel, then solution_ref.solution returns u (if K = nothing) or v (if K <: AbstactMatrix), and s is found form transforming u or v into s using A, B, and K matrices.\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_s0-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_s0","text":"get_s0(LQDynamicData)\nget_s0(SparseLQDynamicModel)\nget_s0(DenseLQDynamicModel)\n\nReturn the value s0 from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_sl-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_sl","text":"get_sl(LQDynamicData)\nget_sl(SparseLQDynamicModel)\nget_sl(DenseLQDynamicModel)\n\nReturn the value sl from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_su-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_su","text":"get_su(LQDynamicData)\nget_su(SparseLQDynamicModel)\nget_su(DenseLQDynamicModel)\n\nReturn the value su from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_u-Union{Tuple{MK}, Tuple{M3}, Tuple{M2}, Tuple{M1}, Tuple{V}, Tuple{T}, Tuple{Any, SparseLQDynamicModel{T, V, M1, M2, M3, MK}}} where {T, V<:AbstractVector{T}, M1<:AbstractMatrix{T}, M2<:AbstractMatrix{T}, M3<:AbstractMatrix{T}, MK<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels.get_u","text":"get_u(solution_ref, lqdm::SparseLQDynamicModel) -> u <: vector\nget_u(solution_ref, lqdm::DenseLQDynamicModel) -> u <: vector\n\nQuery the solution u from the solver. If K = nothing, the solution for u is queried from solution_ref.solution\n\nIf K <: AbstractMatrix, solution_ref.solution returns v, and get_u solves for u using the K matrix (and the A and B matrices if lqdm <: DenseLQDynamicModel)\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_ul-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_ul","text":"get_ul(LQDynamicData)\nget_ul(SparseLQDynamicModel)\nget_ul(DenseLQDynamicModel)\n\nReturn the value ul from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_uu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_uu","text":"get_uu(LQDynamicData)\nget_uu(SparseLQDynamicModel)\nget_uu(DenseLQDynamicModel)\n\nReturn the value uu from LQDynamicData or SparseLQDynamicModel.dynamicdata or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_A!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_A!","text":"set_A!(LQDynamicData, row, col, val)\nset_A!(SparseLQDynamicModel, row, col, val)\nset_A!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry A[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_B!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_B!","text":"set_B!(LQDynamicData, row, col, val)\nset_B!(SparseLQDynamicModel, row, col, val)\nset_B!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry B[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_E!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_E!","text":"set_E!(LQDynamicData, row, col, val)\nset_E!(SparseLQDynamicModel, row, col, val)\nset_E!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry E[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_F!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_F!","text":"set_F!(LQDynamicData, row, col, val)\nset_F!(SparseLQDynamicModel, row, col, val)\nset_F!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry F[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_K!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_K!","text":"set_K!(LQDynamicData, row, col, val)\nset_K!(SparseLQDynamicModel, row, col, val)\nset_K!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry K[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_Q!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_Q!","text":"set_Q!(LQDynamicData, row, col, val)\nset_Q!(SparseLQDynamicModel, row, col, val)\nset_Q!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry Q[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_Qf!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_Qf!","text":"set_Qf!(LQDynamicData, row, col, val)\nset_Qf!(SparseLQDynamicModel, row, col, val)\nset_Qf!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry Qf[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_R!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_R!","text":"set_R!(LQDynamicData, row, col, val)\nset_R!(SparseLQDynamicModel, row, col, val)\nset_R!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry R[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_S!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_S!","text":"set_S!(LQDynamicData, row, col, val)\nset_S!(SparseLQDynamicModel, row, col, val)\nset_S!(DenseLQDynamicModel, row, col, val)\n\nSet the value of entry S[row, col] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_gl!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_gl!","text":"set_gl!(LQDynamicData, index, val)\nset_gl!(SparseLQDynamicModel, index, val)\nset_gl!(DenseLQDynamicModel, index, val)\n\nSet the value of entry gl[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_gu!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_gu!","text":"set_gu!(LQDynamicData, index, val)\nset_gu!(SparseLQDynamicModel, index, val)\nset_gu!(DenseLQDynamicModel, index, val)\n\nSet the value of entry gu[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_s0!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_s0!","text":"set_s0!(LQDynamicData, index, val)\nset_s0!(SparseLQDynamicModel, index, val)\nset_s0!(DenseLQDynamicModel, index, val)\n\nSet the value of entry s0[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_sl!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_sl!","text":"set_sl!(LQDynamicData, index, val)\nset_sl!(SparseLQDynamicModel, index, val)\nset_sl!(DenseLQDynamicModel, index, val)\n\nSet the value of entry sl[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_su!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_su!","text":"set_su!(LQDynamicData, index, val)\nset_su!(SparseLQDynamicModel, index, val)\nset_su!(DenseLQDynamicModel, index, val)\n\nSet the value of entry su[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_ul!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_ul!","text":"set_ul!(LQDynamicData, index, val)\nset_ul!(SparseLQDynamicModel, index, val)\nset_ul!(DenseLQDynamicModel, index, val)\n\nSet the value of entry ul[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_uu!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_uu!","text":"set_uu!(LQDynamicData, index, val)\nset_uu!(SparseLQDynamicModel, index, val)\nset_uu!(DenseLQDynamicModel, index, val)\n\nSet the value of entry uu[index] to val for LQDynamicData, SparseLQDynamicModel.dynamicdata, or DenseLQDynamicModel.dynamicdata\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearOperators.reset!-Union{Tuple{DynamicNLPModels.LQJacobianOperator{T, V, M}}, Tuple{M}, Tuple{V}, Tuple{T}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}","page":"API Manual","title":"LinearOperators.reset!","text":"LinearOperators.reset!(Jac::LQJacobianOperator{T, V, M})\n\nResets the values of attributes SJ1, SJ2, and SJ3 to zero\n\n\n\n\n\n","category":"method"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Welcome to the documentation of DynamicNLPModels.jl","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"warning: Warning\nThis documentation page is under construction.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"note: Note\nThis documentation is also available in PDF format.","category":"page"},{"location":"#What-is-DynamicNLPModels?","page":"Introduction","title":"What is DynamicNLPModels?","text":"","category":"section"},{"location":"#Bug-reports-and-support","page":"Introduction","title":"Bug reports and support","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Please report issues and feature requests via the Github issue tracker. ","category":"page"}]
}
