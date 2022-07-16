using RDatasets, DataFrames, CSV
using Random, Distributions, Distributed
using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD
# using KCenters
using Distances, LinearAlgebra, Statistics
using Printf

using JuMP
using Ipopt, CPLEX#, SCIP

# arg1=: number of clusters to be solved
# arg2=: number of points in a cluster
# arg3=: number of nlines
# arg4=: cplex methods, QCP, LCP, LCP_CUT
# arg5=: thread, true: multi-thread, false: single-thread

# load functions for branch&bound and data preprocess from self-created module

if !("scr/" in LOAD_PATH)
    push!(LOAD_PATH, "scr/")
end
using kcenter_opt, data_process


Random.seed!(1) #120
clst_n = parse(Int, ARGS[2])  # number of points in a cluster 
nclst = 3 # number of clusters that a generated toy-data has
d = 2 # dimension
data = Array{Float64}(undef, d, clst_n*nclst) # initial data array (clst_n*k)*2 
# all toy datasets in the paper are generated randomly with seed 1 
# and generated by the following code with the same parameters
# with different clst_n
mu = reshape(sample(1:30, nclst*d), nclst, d)
for i = 1:nclst::Int
    sig = round.(sig_gen(sample(1:10, d)))
    # println(sig)
    clst = rand(MvNormal(mu[i,:], sig), clst_n) # data is 2*clst_n
    data[:,((i-1)*clst_n+1):(i*clst_n)] = clst
end

println("data size: ", size(data))
println("data type: ", typeof(data))

k = parse(Int, ARGS[1]) #length(unique(label))
Random.seed!(123)

nlines = parse(Int, ARGS[3])

method = ARGS[4]
thread_flag = parse(Bool, ARGS[5])
if method == "LCP_CUT" # callback cut
    t_g = @elapsed centers_g, objv_g, gap_g, cplex_cost, node_number = global_OPT_L4_CPX(data, k, nlines, thread_flag)
elseif method == "LCP" 
    t_g = @elapsed centers_g, objv_g, gap_g, cplex_cost, node_number = global_OPT_L(data, k, nlines, thread_flag)
elseif method == "QCP"
    t_g = @elapsed centers_g, objv_g, gap_g, cplex_cost, node_number = global_OPT_x(data, k, thread_flag)
end

println("$clst_n:\t", round(objv_g, digits=2), "\t", round(cplex_cost, digits=2), "\t", round(t_g, digits=2), "s\t", round(gap_g, digits=4), "%\t", node_number)
