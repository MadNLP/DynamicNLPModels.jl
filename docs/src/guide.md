
# Getting Started

DynamicNLPModels.jl takes user defined data to construct a linear MPC problem of the form

```math
\begin{aligned}
\min_{s, u, v} &\; s_N^\top Q_f s_N + \frac{1}{2} \sum_{i = 0}^{N-1} \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]^\top \left[ \begin{array}{cc} Q & S \\ S^\top & R \end{array} \right] \left[ \begin{array}{c} s_i \\ u_i \end{array} \right]\\
          \textrm{s.t.} &\;s_{i+1} = As_i + Bu_i + w_i \quad \forall i = 0, 1, \cdots, N - 1 \\
          &\; u_i = Ks_i + v_i \quad  \forall i = 0, 1, \cdots, N - 1 \\
          &\; g^l \le E s_i + F u_i \le g^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s^l \le s_i \le s^u \quad \forall i = 0, 1, \cdots, N \\
          &\; u^l \le u_i \le u^u \quad \forall i = 0, 1, \cdots, N - 1\\
          &\; s_0 = \bar{s}.
\end{aligned}
```

This data is stored within the struct `LQDynamicData`, which can be created by passing the data `s0`, `A`, `B`, `Q`, `R` and `N` to the constructor as in the example below. 

```julia
using DynamicNLPModels, Random, LinearAlgebra

Q  = 1.5 * Matrix(I, (3, 3))
R  = 2.0 * Matrix(I, (2, 2))
A  = rand(3, 3)
B  = rand(3, 2)
N  = 5
s0 = [1.0, 2.0, 3.0]

lqdd = LQDynamicData(s0, A, B, Q, R, N; **kwargs)
```

`LQDynamicData` contains the following fields. All fields after `R` are keyword arguments:
 * `ns`: number of states (determined from size of `Q`)
 * `nu`: number of inputs (determined from size of `R`)
 * `N` : number of time steps
 * `s0`: a vector of initial states
 * `A` : matrix that is multiplied by the states that corresponds to the dynamics of the problem. Number of columns is equal to `ns`
 * `B` : matrix that is multiplied by the inputs that corresonds to the dynamics of the problem. Number of columns is equal to `nu`
 * `Q` : objective function matrix for system states from ``0, 1, \cdots, (N - 1)``
 * `R` : objective function matrix for system inputs from ``0, 1, \cdots, (N - 1)``
 * `Qf`: objective function matrix for system states at time ``N``
 * `S` : objective function matrix for system states and inputs
 * `E` : constraint matrix multiplied by system states. Number of columns is equal to `ns`
 * `F` : constraint matrix multiplied by system inputs. Number of columns is equal to `nu`
 * `K` : feedback gain matrix. Used to ensure numerical stability of the condensed problem. Not necessary within the sparse problem
 * `w` : constant term within dynamic constraints. At this time, this is the only data that is time varying. This vector must be length `ns` * `N`, where each set of `ns` entries corresponds to that time (i.e., entries `1:ns` correspond to time ``0``, entries `(ns + 1):(2 * ns)` corresond to time ``1``, etc.)
 * `sl` : lower bounds on state variables
 * `su` : upper bounds on state variables 
 * `ul` : lower bounds on ipnut variables
 * `uu` : upper bounds on input variables
 * `gl` : lower bounds on the constraints ``Es_i + Fu_i``
 * `gu` : upper bounds on the constraints ``Es_i + Fu_i``

## `SparseLQDynamicModel`

A `SparseLQDynamicModel` can be created by either passing `LQDynamicData` to the constructor or passing the data itself, where the same keyword options exist which can be used for `LQDynamicData`. 

```julia 
sparse_lqdm = SparseLQDynamicModel(lqdd)

# or 

sparse_lqdm = SparseLQDynamicModel(s0, A, B, Q, R, N; **kwargs)
```

The `SparseLQDynamicModel` contains four fields:
 * `dynamic_data` which contains the `LQDynamicData`
 * `data` which is the `QPData` from [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl). This object also contains the following data:
   - `H` which is the Hessian of the linear MPC problem 
   - `A` which is the Jacobian of the linear MPC problem such that ``\textrm{lcon} \le A z \le \textrm{ucon}``
   - `c` which is the linear term of a quadratic objective function
   - `c0` which is the constant term of a quadratic objective function
 * `meta` which contains the `NLPModelMeta` for the problem from NLPModels.jl
 * `counters` which is the `Counters` object from NLPModels.jl

!!! note
    The `SparseLQDynamicModel` requires that all matrices in the `LQDynamicData` be the same type. It is recommended that the user be aware of how to most efficiently store their data in the `Q`, `R`, `A`, and `B` matrices as this impacts how efficiently the `SparseLQDynamicModel` is constructed. When `Q`, `R`, `A`, and `B` are sparse, building the `SparseLQDynamicModel` is much faster when these are passed as sparse rather than dense matrices. 

## `DenseLQDynamicModel`

The `DenseLQDynamicModel` eliminates the states within the linear MPC problem to build an equivalent optimization problem that is only a function of the inputs. This can be particularly useful when the number of states is large compared to the number of inputs.

A `DenseLQDynamicModel` can be created by either passing `LQDynamicData` to the constructor or passing the data itself, where the same keyword options exist which can be used for `LQDynamicData`. 

```julia 
dense_lqdm = DenseLQDynamicModel(lqdd)

# or 

dense_lqdm = DenseLQDynamicModel(s0, A, B, Q, R, N; **kwargs)
```

The `DenseLQDynamicModel` contains five fields:
 * `dynamic_data` which contains the `LQDynamicData`
 * `data` which is the `QPData` from [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl). This object also contains the following data:
   - `H` which is the Hessian of the condensed linear MPC problem 
   - `A` which is the Jacobian of the condensed linear MPC problem such that ``\textrm{lcon} \le A z \le \textrm{ucon}``
   - `c` which is the linear term of the condensed linear MPC problem 
   - `c0` which is the constant term of the condensed linear MPC problem
 * `meta` which contains the `NLPModelMeta` for the problem from NLPModels.jl
 * `counters` which is the `Counters` object from NLPModels.jl
 * `blocks` which contains the data needed to condense the model and then to update the condensed model when `s0` is reset. 

The `DenseLQDynamicModel` is formed from dense matrices, and this dense system can be solved on a GPU using MadNLP.jl and MadNLPGPU.jl For an example script for performing this, please see the the [examples directory](https://github.com/MadNLP/DynamicNLPModels.jl/tree/main/examples) of the main repository.

## API functions
An API has been created for working with `LQDynamicData` and the sparse and dense models. All functions can be seen in the API Manual section. However, we give a short overview of these functions here. 

 * `reset_s0!(LQDynamicModel, new_s0)`: resets the model in place with a new `s0` value. This could be called after each sampling period in MPC to reset the model with a new measured value
 * `get_s(solver_ref, LQDynamicModel)`: returns the optimal solution for the states from a given solver reference
 * `get_u(solver_ref, LQDynamicModel)`: returns the optimal solution for the inputs from a given solver reference; when `K` is defined, the solver reference contains the optimal ``v`` values rather than optimal ``u`` values, adn this function converts ``v`` to ``u`` and returns the ``u`` values
 * `get_*`:  returns the data of `*` where `*` is an object within `LQDynamicData`
 * `set_*!`: sets the value within the data of `*` for a given entry to a user defined value
 