var documenterSearchIndex = {"docs":
[{"location":"api/#API-Manual","page":"API Manual","title":"API Manual","text":"","category":"section"},{"location":"api/","page":"API Manual","title":"API Manual","text":"Modules = [DynamicNLPModels]","category":"page"},{"location":"api/#DynamicNLPModels.LQDynamicData","page":"API Manual","title":"DynamicNLPModels.LQDynamicData","text":"LQDynamicData{T,V,M} <: AbstractLQDynData{T,V}\n\nA struct to represent the features of the optimization problem \n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\nAttributes include:\n\ns0: initial state of system\nA : constraint matrix for system states\nB : constraint matrix for system inputs\nQ : objective function matrix for system states from 1:(N-1)\nR : objective function matrix for system inputs from 1:(N-1)\nN : number of time steps\nQf: objective function matrix for system state at time N\nS : objective function matrix for system states and inputs\nns: number of state variables\nnu: number of input varaibles\nE : constraint matrix for state variables\nF : constraint matrix for input variables\nsl: vector of lower bounds on state variables\nsu: vector of upper bounds on state variables\nul: vector of lower bounds on input variables\nuu: vector of upper bounds on input variables\ngl: vector of lower bounds on constraints\ngu: vector of upper bounds on constraints\n\nsee also LQDynamicData(s0, A, B, Q, R, N; ...)\n\n\n\n\n\n","category":"type"},{"location":"api/#DynamicNLPModels.LQDynamicData-Union{Tuple{M}, Tuple{V}, Tuple{T}, Tuple{V, M, M, M, M, Any}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels.LQDynamicData","text":"LQDynamicData(s0, A, B, Q, R, N; ...) -> LQDynamicData{T, V, M}\n\nA constructor for building an object of type LQDynamicData for the optimization problem \n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1\n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\ns0: initial state of system\nA : constraint matrix for system states\nB : constraint matrix for system inputs\nQ : objective function matrix for system states from 1:(N-1)\nR : objective function matrix for system inputs from 1:(N-1)\nN : number of time steps\n\nThe following attributes of the LQDynamicData type are detected automatically from the length of s0 and size of R\n\nns: number of state variables\nnu: number of input varaibles\n\nThe following keyward arguments are also accepted\n\nQf = Q: objective function matrix for system state at time N; dimensions must be ns x ns\nS  = nothing: objective function matrix for system state and inputs\nE  = zeros(0, ns) : constraint matrix for state variables\nF  = zeros(0, nu) : constraint matrix for input variables\nsl = fill(-Inf, ns): vector of lower bounds on state variables\nsu = fill(Inf, ns) : vector of upper bounds on state variables\nul = fill(-Inf, nu): vector of lower bounds on input variables\nuu = fill(Inf, nu) : vector of upper bounds on input variables\ngl = fill(-Inf, size(E, 1)) : vector of lower bounds on constraints\ngu = fill(Inf, size(E, 1))  : vector of upper bounds on constraints\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.LQDynamicModel-Union{Tuple{LQDynamicData{T, V, M}}, Tuple{M}, Tuple{V}, Tuple{T}} where {T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}","page":"API Manual","title":"DynamicNLPModels.LQDynamicModel","text":"LQDynamicModel(dnlp::LQDynamicData; condense=false)      -> LQdynamicModel\nLQDynamicModel(s0, A, B, Q, R, N; condense = false, ...) -> LQDynamicModel\n\nA constructor for building a LQDynamicModel <: QuadraticModels.AbstractQuadraticModel from LQDynamicData Input data is for the problem of the form \n\n    minimize    frac12 sum_i = 0^N-1(s_i^T Q s_i + 2 u_i^T S^T x_i + u_i^T R u_i) + frac12 s_N^T Qf s_N\n    subject to  s_i+1 = A s_i + B u_i  for i=0 1  N-1\n                gl le E s_i + F u_i le gu for i = 0 1  N-1            \n                sl le s le su\n                ul le u le uu\n                s_0 = s0\n\n\n\nIf condense=false, data is converted to the form \n\n    minimize    frac12 z^T H z \n    subject to  lcon le Jz le ucon\n                lvar le z le uvar\n\nResulting H and J matrices are stored as QuadraticModels.QPData within the LQDynamicModel struct and  variable and constraint limits are stored within NLPModels.NLPModelMeta\n\n\n\nIf condense=true, data is converted to the form \n\n    minimize    frac12 u^T H u + h^T u + h0 \n    subject to  Jz le g\n                ul le u le uu\n\nResulting H, J, h, and h0 matrices are stored within QuadraticModels.QPData as H, A, c, and c0 attributes respectively\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels._build_H-Union{Tuple{M}, Tuple{M, M, Any}} where M<:(AbstractMatrix)","page":"API Manual","title":"DynamicNLPModels._build_H","text":"_build_H(Q, R, N; Qf = []) -> H\n\nBuild the (sparse) H matrix from square Q and R matrices such that   z^T H z = sum{i=1}^{N-1} si^T Q s + sum{i=1}^{N-1} u^T R u + sN^T Qf s_n . \n\nExamples\n\njulia> Q = [1 2; 2 1]; R = ones(1,1); _build_H(Q, R, 2)\n6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:\n 1.0  2.0   ⋅    ⋅    ⋅    ⋅ \n 2.0  1.0   ⋅    ⋅    ⋅    ⋅\n  ⋅    ⋅   1.0  2.0   ⋅    ⋅\n  ⋅    ⋅   2.0  1.0   ⋅    ⋅\n  ⋅    ⋅    ⋅    ⋅   1.0   ⋅\n  ⋅    ⋅    ⋅    ⋅    ⋅     ⋅\n\nIf Qf is not given, then Qf defaults to Q\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels._build_sparse_J1-Union{Tuple{M}, Tuple{M, M, Any}} where M<:(AbstractMatrix)","page":"API Manual","title":"DynamicNLPModels._build_sparse_J1","text":"_build_sparse_J1(A, B, N) -> J\n\nBuild the (sparse) J matrix or a linear model from A and B matrices such that 0 <= Jz <= 0 is equivalent to s{i+1} = Asi + Bs_i for i = 1,..., N-1\n\nExamples\n\njulia> A = [1 2 ; 3 4]; B = [5 6; 7 8]; _build_J(A,B,3)\n4×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 20 stored entries:\n 1.0  2.0  -1.0    ⋅     ⋅     ⋅   5.0  6.0   ⋅    ⋅    ⋅    ⋅\n 3.0  4.0    ⋅   -1.0    ⋅     ⋅   7.0  8.0   ⋅    ⋅    ⋅    ⋅\n  ⋅    ⋅    1.0   2.0  -1.0    ⋅    ⋅    ⋅   5.0  6.0   ⋅    ⋅\n  ⋅    ⋅    3.0   4.0    ⋅   -1.0   ⋅    ⋅   7.0  8.0   ⋅    ⋅\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_A-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_A","text":"get_A(LQDynamicData)\nget_A(LQDynamicModel)\n\nReturn the value A from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_B-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_B","text":"get_B(LQDynamicData)\nget_B(LQDynamicModel)\n\nReturn the value B from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_E-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_E","text":"get_E(LQDynamicData)\nget_E(LQDynamicModel)\n\nReturn the value E from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_F-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_F","text":"get_F(LQDynamicData)\nget_F(LQDynamicModel)\n\nReturn the value F from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_K-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_K","text":"get_K(LQDynamicData)\nget_K(LQDynamicModel)\n\nReturn the value K from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_N-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_N","text":"get_N(LQDynamicData)\nget_N(LQDynamicModel)\n\nReturn the value N from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_Q-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_Q","text":"get_Q(LQDynamicData)\nget_Q(LQDynamicModel)\n\nReturn the value Q from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_Qf-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_Qf","text":"get_Qf(LQDynamicData)\nget_Qf(LQDynamicModel)\n\nReturn the value Qf from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_R-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_R","text":"get_R(LQDynamicData)\nget_R(LQDynamicModel)\n\nReturn the value R from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_S-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_S","text":"get_S(LQDynamicData)\nget_S(LQDynamicModel)\n\nReturn the value S from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_gl-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_gl","text":"get_gl(LQDynamicData)\nget_gl(LQDynamicModel)\n\nReturn the value gl from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_gu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_gu","text":"get_gu(LQDynamicData)\nget_gu(LQDynamicModel)\n\nReturn the value gu from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_ns-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_ns","text":"get_ns(LQDynamicData)\nget_ns(LQDynamicModel)\n\nReturn the value ns from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_nu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_nu","text":"get_nu(LQDynamicData)\nget_nu(LQDynamicModel)\n\nReturn the value nu from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_s0-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_s0","text":"get_s0(LQDynamicData)\nget_s0(LQDynamicModel)\n\nReturn the value s0 from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_sl-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_sl","text":"get_sl(LQDynamicData)\nget_sl(LQDynamicModel)\n\nReturn the value sl from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_su-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_su","text":"get_su(LQDynamicData)\nget_su(LQDynamicModel)\n\nReturn the value su from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_ul-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_ul","text":"get_ul(LQDynamicData)\nget_ul(LQDynamicModel)\n\nReturn the value ul from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.get_uu-Tuple{LQDynamicData}","page":"API Manual","title":"DynamicNLPModels.get_uu","text":"get_uu(LQDynamicData)\nget_uu(LQDynamicModel)\n\nReturn the value uu from LQDynamicData or LQDynamicModel.dynamic_data\n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_A!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_A!","text":"set_A!(LQDynamicData, row, col, val)\nset_A!(LQDynamicModel, row, col, val)\n\nSet the value of entry A[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_B!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_B!","text":"set_B!(LQDynamicData, row, col, val)\nset_B!(LQDynamicModel, row, col, val)\n\nSet the value of entry B[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_Q!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_Q!","text":"set_Q!(LQDynamicData, row, col, val)\nset_Q!(LQDynamicModel, row, col, val)\n\nSet the value of entry Q[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_Qf!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_Qf!","text":"set_Qf!(LQDynamicData, row, col, val)\nset_Qf!(LQDynamicModel, row, col, val)\n\nSet the value of entry Qf[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_R!-Tuple{LQDynamicData, Any, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_R!","text":"set_R!(LQDynamicData, row, col, val)\nset_R!(LQDynamicModel, row, col, val)\n\nSet the value of entry R[row, col] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_s0!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_s0!","text":"set_s0!(LQDynamicData, index, val)\nset_s0!(LQDynamicModel, index, val)\n\nSet the value of entry s0[index] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_sl!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_sl!","text":"set_sl!(LQDynamicData, index, val)\nset_sl!(LQDynamicModel, index, val)\n\nSet the value of entry sl[index] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_su!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_su!","text":"set_su!(LQDynamicData, index, val)\nset_su!(LQDynamicModel, index, val)\n\nSet the value of entry su[index] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_ul!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_ul!","text":"set_ul!(LQDynamicData, index, val)\nset_ul!(LQDynamicModel, index, val)\n\nSet the value of entry ul[index] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"api/#DynamicNLPModels.set_uu!-Tuple{LQDynamicData, Any, Any}","page":"API Manual","title":"DynamicNLPModels.set_uu!","text":"set_uu!(LQDynamicData, index, val)\nset_uu!(LQDynamicModel, index, val)\n\nSet the value of entry uu[index] to val for LQDynamicData or LQDynamicModel.dynamic_data \n\n\n\n\n\n","category":"method"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Welcome to the documentation of DynamicNLPModels.jl","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"warning: Warning\nThis documentation page is under construction.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"note: Note\nThis documentation is also available in PDF format.","category":"page"},{"location":"#What-is-DynamicNLPModels?","page":"Introduction","title":"What is DynamicNLPModels?","text":"","category":"section"},{"location":"#Bug-reports-and-support","page":"Introduction","title":"Bug reports and support","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Please report issues and feature requests via the Github issue tracker. ","category":"page"}]
}
