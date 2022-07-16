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
# arg2=: dataset name
# arg3=: number of nlines
# arg4=: cplex methods, QCP, LCP, LCP_CUT
# arg5=: thread, true: multi-thread, false: single-thread

# load functions for branch&bound and data preprocess from self-created module

if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end
using kcenter_opt, data_process

dataname = ARGS[2]
if dataname == "iris"
    data = data_preprocess("iris") # read iris data from datasets package
else
    if Sys.iswindows()
        data = data_preprocess(dataname, nothing, joinpath(@__DIR__, "..\\data\\"), "NA") # read data in Windows
    else
        data = data_preprocess(dataname, nothing, joinpath(@__DIR__, "../data/"), "NA") # read data in Mac
    end
end
println("data size: ", size(data))
println("data type: ", typeof(data))

k = parse(Int, ARGS[1]) 
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

println("$dataname:\t", round(objv_g, digits=2), "\t", round(cplex_cost, digits=2), "\t", round(t_g, digits=2), "s\t", round(gap_g, digits=4), "%\t", node_number)

