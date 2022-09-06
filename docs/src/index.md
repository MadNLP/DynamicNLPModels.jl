# Introduction
[DynamicNLPModels.jl](https://github.com/MadNLP/DynamicNLPModels.jl) is a package for [Julia](https://julialang.org/) designed for representing linear [model predictive control (MPC)](https://en.wikipedia.org/wiki/Model_predictive_control) problems. It includes an API for building a model from user defined data and querying solutions.

!!! note
	This documentation is also available in [PDF format](DynamicNLPModels.pdf).
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

```math
\begin{aligned}
\min_{s, u, v} &\; s_N^\top Q_f s_N + \frac{1}{2} \sum_{i = 0}^{N-1} \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]^\top \left[ \begin{array}{cc} Q & S \\ S^\top & R \end{array} \right] \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]\\
          \textrm{s.t.} &\;s_{i+1} = As_i + Bu_i + w_i \quad \forall i = 0, 1, \cdots, N - 1 \\
          &\; u_i = Ks_i + v_i \quad  \forall i = 0, 1, \cdots, N - 1 \\
          &\; g^l \le E s_i + F u_i \le g^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s^l \le s_i \le s^u \quad \forall i = 0, 1, \cdots, N \\
          &\; u^l \le u_t \le u^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s_0 = \bar{s} 
\end{aligned}
```

where ``s_i`` are the states, ``u_i`` are the inputs, ``N`` is the time horizon, ``\bar{s}`` are the initial states, and ``Q``, ``R``, ``A``, and ``B`` are user defined data. The matrices ``Q_f``, ``S``, ``K``, ``E``, and ``F`` and the vectors ``w``, ``g^l``, ``g^u``, ``s^l``, ``s^u``, ``u^l``, and ``u^u`` are optional data. ``v_t`` is only needed in the condensed formulation, and it arises when ``K`` is defined by the user to ensure numerical stability of the condensed problem. 

The condensed formulation used within DynamicNLPModels.jl is 

```math
\begin{aligned}
\min_{\boldsymbol{v}} &\;\; \frac{1}{2} \boldsymbol{v}^\top \boldsymbol{H} \boldsymbol{v} + \boldsymbol{h}^\top \boldsymbol{v} + \boldsymbol{h}_0\\
        \textrm{s.t.} &\; d^l \le \boldsymbol{J} \boldsymbol{v} \le d^u.
\end{aligned}
```


# Bug reports and support
This package is new and still undergoing some development. If you encounter a bug, please report it through Github's [issue tracker](https://github.com/MadNLP/DynamicNLPModels.jl/issues). 