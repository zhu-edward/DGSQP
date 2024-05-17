# Dynamic Game SQP

Author: Edward Zhu (edward.zhu@berkeley.edu)

This repository contains a Python implementation of the Dynamic Game SQP algorithm.

## Related Papers
- Zhu, Edward L., and Francesco Borrelli. "A sequential quadratic programming approach to the solution of open-loop generalized nash equilibria." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023. [arXiv](http://arxiv.org/abs/2203.16478)
- Zhu, Edward L., and Francesco Borrelli. "A Sequential Quadratic Programming Approach to the Solution of Open-Loop Generalized Nash Equilibria for Autonomous Racing." arXiv preprint arXiv:2404.00186 (2024). (Under review). [arXiv](https://arxiv.org/abs/2404.00186), [data](https://drive.google.com/drive/folders/1VKuO1ntuB2cQMq-e8TyQafEe6IZEyAos?usp=sharing)

## Installation
Dependencies:
- `numpy`
- `scipy`
- `casadi`
- `osqp`
- `matplotlib`

The following command will install the Python package `DGSQP`:
```
python3 -m pip install -e .
```
### Installing the CPLEX QP solver for DGSQP

CPLEX is a QP, LP, and mixed integer solver developed and maintained by IBM and is provided as a part of [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/products/ilog-cplex-optimization-studio). Academic users are allowed free access to the solver after registering for an account. To use CPLEX with CasADi Python:
1. Download ILOG CPLEX Optimization Studio according to https://ronennir.medium.com/installing-cplex-optimization-studio-on-ubuntu-20-04-53e234ca4ec2.
2. Install CPLEX using `sudo` (this will default to `/opt/ibm/ILOG` on Ubuntu).
3. Copy the compiled CPLEX library (found at `/opt/ibm/ILOG/CPLEX_Studio<CPLEX_VERSION>/cplex/bin/x86-64_linux/libcplex<CPLEX_VERSION>.so`), to the CasADi Python package installation directory:
   - Local installation: `~/.local/lib/python3.8/site-packages/casadi`
   - Global installation: `/usr/local/lib/python3.8/site-packages/casadi`
4. Before creating the solver (e.g. using `ca.conic`), set the system environment variable `CPLEX_VERSION` to match the version of the solver that you have installed (e.g. using `os.environ['CPLEX_VERSION']='2210'` for v22.1.0).
5. Set the parameter `DGSQPV2Params.qp_solver = 'cplex'`

### Installing the PATH solver

To run the comparisons using the PATH solver, install the prerequisites using the following steps
1. Install Julia:
- `wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz`
- `tar zxvf julia-1.10.1-linux-x86_64.tar.gz`
- `export PATH="$PATH:/path/to/<Julia directory>/bin"`
2. Install Julia packages:
- In the Julia REPL package manager: 
- `add PyCall`
- `add PATHSolver@1.1.1` 
- (note, only version 1.1.1 works when called from pyjulia)

3. Install pyjulia:
- `python3 -m pip install julia`

## Quick Start
Scripts for running the experiments presented in the paper can be found in the `scripts` directory.
