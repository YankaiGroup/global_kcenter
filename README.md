# Global K-Center
The global branch and bound algorithm for K-Center Clustering.

## Related Paper
Shi, M., Hua, K., Ren, J., & Cao, Y. (2022, June). [Global Optimization of K-Center Clustering](https://proceedings.mlr.press/v162/shi22b.html). In International Conference on Machine Learning (pp. 19956-19966). PMLR.

## Prerequisite Packages
* Julia - 1.6.3 or above
* CPLEX - 20.1.0
* Julia Packages
    * Distributions, LinearAlgebra, RDatasets, DataFrames, CSV, CategoricalArrays, Clustering, Random, Distances, StatsBase, Printf, Statistics, MLDataUtils, SharedArrays, JuMP, Ipopt, CPLEX
## Test Cases

``` bash
# BB+CF
julia test/kcenter_real.jl 3 iris false false
julia test/kcenter_real.jl 3 rng_agr false false 
julia test/kcenter_real.jl 3 syn_300 false false

# BB+CF+FBBT 
julia test/kcenter_real.jl 3 iris true false
julia test/kcenter_real.jl 3 rng_agr true false 
julia test/kcenter_real.jl 3 syn_300 true false

# CPLEX+Q
julia test/kcenter_cplex_L.jl 3 iris 3 QCP true 
julia test/kcenter_cplex_L.jl 3 rng_agr 3 QCP true 
julia test/kcenter_cplex_L.jl 3 syn_300 3 QCP true 

# CPLEX+L3
julia test/kcenter_cplex_L.jl 3 iris 3 LCP true 
julia test/kcenter_cplex_L.jl 3 rng_agr 3 LCP true 
julia test/kcenter_cplex_L.jl 3 syn_300 3 LCP true 

# CPLEX+L3_CUT
julia test/kcenter_cplex_L.jl 3 iris 3 LCP_CUT false
julia test/kcenter_cplex_L.jl 3 rng_agr 3 LCP_CUT false
julia test/kcenter_cplex_L.jl 3 syn_300 3 LCP_CUT false
```
