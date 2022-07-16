
Please refer to [github](https://github.com/YankaiGroup/Kcenter ) for more details. 

## Test Cases
``` bash
# BB+CF
julia test/kcenter_real.jl 3 iris false false

# BB+CF+FBBT 
julia test/kcenter_real.jl 3 iris true false

# CPLEX+Q
## realworld datasets
julia test/kcenter_cplex_L.jl 3 iris 3 QCP true 
## sythetic datasets
julia test/kcenter_cplex_s.jl 3 100 3 QCP true 

# CPLEX+L3
## realworld datasets
julia test/kcenter_cplex_L.jl 3 iris 3 LCP true 
## sythetic datasets
julia test/kcenter_cplex_s.jl 3 100 3 LCP true 

# CPLEX+L3_CUT
## realworld datasets
julia test/kcenter_cplex_L.jl 3 iris 3 LCP_CUT false
## sythetic datasets
julia test/kcenter_cplex_s.jl 3 100 3 LCP_CUT false 
```