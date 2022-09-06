# DynamicNLPModels.jl

| **Documentation** | **Build Status** | **Coverage** |
|:-----------------:|:----------------:|:----------------:|
| [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://madnlp.github.io/DynamicNLPModels.jl/dev) | [![build](https://github.com/MadNLP/DynamicNLPModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MadNLP/DynamicNLPModels.jl/actions) | [![codecov](https://codecov.io/gh/MadNLP/DynamicNLPModels.jl/branch/main/graph/badge.svg?token=2Z18FIU4R7)](https://codecov.io/gh/MadNLP/DynamicNLPModels.jl) |

DynamicNLPModels.jl is a package for [Julia](https://julialang.org/) designed for representing linear [model predictive control (MPC)](https://en.wikipedia.org/wiki/Model_predictive_control) problems. It includes an API for building a model from user defined data and querying solutions.


## Installation

To install this package, please use

```julia 
using Pkg
Pkg.add(url="https://github.com/MadNLP/DynamicNLPModels.jl.git")
```

or

```julia
pkg> add https://github.com/MadNLP/DynamicNLPModels.jl.git
```

## Overview

DynamicNLPModels.jl can construct both sparse and condensed formulations for MPC problems based on user defined data. We use the methods discussed by [Jerez et al.](https://doi.org/10.1016/j.automatica.2012.03.010) to eliminate the states and condense the problem. DynamicNLPModels.jl constructs models that are subtypes of `AbstractNLPModel` from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) enabling both the sparse and condensed models to be solved with a variety of different solver packages in Julia. DynamicNLPModels was designed in part with the goal of solving linear MPC problems on the GPU. This can be done within [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl) using [MadNLPGPU.jl](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPGPU). 

The general sparse formulation used within DynamicNLPModels.jl is

$$\begin{align*}
\min_{s, u, v} &\; s_N^\top Q_f s_N + \frac{1}{2} \sum_{i = 0}^{N-1} \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]^\top \left[ \begin{array}{cc} Q & S \\ S^\top & R \end{array} \right] \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]\\
          \textrm{s.t.} &\;s_{i+1} = As_i + Bu_i + w_i \quad \forall i = 0, 1, \cdots, N - 1 \\
          &\; u_i = Ks_i + v_i \quad  \forall i = 0, 1, \cdots, N - 1 \\
          &\; g^l \le E s_i + F u_i \le g^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s^l \le s_i \le s^u \quad \forall i = 0, 1, \cdots, N \\
          &\; u^l \le u_i \le u^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s_0 = \bar{s} 
\end{align*}$$

where $s_i$ are the states, $u_i$ are the inputs$, $N$ is the time horizon, $\bar{s}$ are the initial states, and $Q$, $R$, $A$, and $B$ are user defined data. The matrices $Q_f$, $S$, $K$, $E$, and $F$ and the vectors $w$, $g^l$, $g^u$, $s^l$, $s^u$, $u^l$, and $u^u$ are optional data. $v_t$ is only needed in the condensed formulation, and it arises when $K$ is defined by the user to ensure numerical stability of the condensed problem. 

The condensed formulation used within DynamicNLPModels.jl is 

$$\begin{align*}
\min_{\boldsymbol{v}} &\;\; \frac{1}{2} \boldsymbol{v}^\top \boldsymbol{H} \boldsymbol{v} + \boldsymbol{h}^\top \boldsymbol{v} + \boldsymbol{h}_0\\
        \textrm{s.t.} &\; d^l \le \boldsymbol{J} \boldsymbol{v} \le d^u.
\end{align*}$$

## Getting Started

DynamicNLPModels.jl takes user defined data to form a `SparseLQDyanmicModel` or a `DenseLQDynamicModel`. The user can first create an object containing the `LQDynamicData`, or they can pass the data directly to the `SparseLQDynamicModel` or `DenseLQDynamicModel` constructors.

```julia
using DynamicNLPModels, Random, LinearAlgebra

Q  = 1.5 * Matrix(I, (3, 3))
R  = 2.0 * Matrix(I, (2, 2))
A  = rand(3, 3)
B  = rand(3, 2)
N  = 5
s0 = [1.0, 2.0, 3.0]

lqdd = LQDynamicData(s0, A, B, Q, R, N; **kwargs)

sparse_lqdm = SparseLQDynamicModel(lqdd)
dense_lqdm  = DenseLQDynamicModel(lqdd)

# or 

sparse_lqdm = SparseLQDynamicModel(s0, A, B, Q, R, N; **kwargs)
dense_lqdm  = DenseLQDynamicModel(s0, A, B, Q, R, N; **kwargs)
```

Optional data (such as $s^l$, $s^u$, $S$, or $Q_f$) can be passed as key word arguments. The models `sparse_lqdm` or `dense_lqdm` can be solved by different solvers such as MadNLP.jl or Ipopt (Ipopt requires the extension NLPModelsIpopt.jl). An example script under `\examples` shows how the dense problem can be solved on a GPU using MadNLPGPU.jl. 

DynamicNLPModels.jl also includes an API for querying solutions and reseting data. Solutions can be queried using `get_u(solver_ref, dynamic_model)` and `get_s(solver_ref, dynamic_model)`. The problem can be reset with a new $s_0$ by calling `reset_s0!(dynamic_model, s0)`. 

