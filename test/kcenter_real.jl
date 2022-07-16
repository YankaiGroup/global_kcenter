using RDatasets, DataFrames, CSV
using Random, Distributions, Distributed
using MLDataUtils, Clustering
using JLD

using TimerOutputs: @timeit, get_timer

# load functions for branch&bound and data preprocess from self-created module
if !("scr/" in LOAD_PATH)
    push!(LOAD_PATH, "scr/")
end
using kcenter_opt, data_process, kcenter_bb

# arg1=: number of clusters to be solved
# arg2=: dataset name
# arg3=: fbbt flag: true, false
# arg4=: symmetric breaking flag: true, false


#############################################################
################# Main Process Program Body #################
#############################################################
const to = get_timer("Shared")

# real world dataset testing
@timeit to "load data" begin
    dataname = ARGS[2]
    if dataname == "iris"
        data, label = data_preprocess("iris") # read iris data from datasets package
    else
        if Sys.iswindows()
            data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "..\\data\\"), "NA") # read data in Windows
        else
            data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "../data/"), "NA") # read data in Mac
        end
    end
    label = vec(label)
    k = parse(Int, ARGS[1])
    Random.seed!(123)
    println("data size: ", size(data))
    println("data type: ", typeof(data))
end # end of @timeit to "load data"

@timeit to "Total BB" begin
    flag_fbbt = parse(Bool, ARGS[3])
    flag_SB = parse(Bool, ARGS[4])
    if flag_fbbt
        if !flag_SB
            # closed form with FBBT, with/without symmetric breaking
            t_FBBT = @elapsed centers, objv, calcInfo = kcenter_bb.branch_bound(data, k, "FBBT", 0) 
        else
            println("wrong parameters")
        end
    else
        if flag_SB
           # closed form with symmetric breaking, without FBBT
           t_sb = @elapsed centers, objv, calcInfo = kcenter_bb.branch_bound(data, k, nothing, 1) 
        else
            # closed without FBBT, without symmetric breaking
            t = @elapsed centers, objv, calcInfo = kcenter_bb.branch_bound(data, k, nothing, 0)  
        end
    end
end # end of @timeit to "Total BB"

# show(to)
# print("\n")


