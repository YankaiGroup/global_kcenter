module kcenter_lb

using Statistics
using Distributions: minimum, maximum
using RDatasets, Random, Distributions
using LinearAlgebra
using MLDataUtils, Clustering #, MLBase
using kcenter_opt, kcenter_ub
using Distributed, SharedArrays
using fbbt


tol = 1e-6
mingap = 1e-3

# function to calcuate the median value of a vector
function med(a,b,c)
    return a+b+c-max(a,b,c)-min(a,b,c)
end

function getGlobalLowerBound(nodeList) # if LB same, choose the first smallest one
    LB = 1e15
    nodeid = 1
    for (idx,n) in enumerate(nodeList)
    	#println("remaining ", idx,  "   ", n.LB)
        if n.LB < LB
            LB = n.LB
            nodeid = idx
        end
    end
    return LB, nodeid
end


function sel_set(X, k, lower, upper)  # select center candidates in the range of each center
    d, n = size(X)
    lwr = zeros(d,k)
    upr = zeros(d,k)

    #all_set = Vector{Any}(undef, k)
    all_num = zeros(k) # all_num is the number of samples that between node.lower and node.upper
    for clst in 1:k
        set = []
        for s in 1:n
            if sum(view(X,:,s) .>= view(lower,:,clst))==d && sum(view(X,:,s) .<= view(upper,:,clst))==d
                push!(set, s)
                all_num[clst] +=1
                if  all_num[clst] == 1
                    lwr[:, clst]=X[:, s]
                    upr[:, clst]=X[:, s]
                elseif all_num[clst] >1
                    for i in 1:d
                        if lwr[i,clst]>X[i, s]
                            lwr[i, clst]=X[i, s]
                        end
                        if upr[i, clst]<X[i,s]
                            upr[i, clst]=X[i, s]
                        end
                    end
                end
            end
        end
        if all_num[clst] == 0     
            lwr = nothing
            upr = nothing
            break
        end
        #all_set[clst] = set
    end
    return lwr, upr    # return set of set, length(all_set)=k
end


function getLB(V, k, d, lwr, upr)
    s_LB = zeros(k) # all distances between V and centers
    all_mu = zeros(d, k)
    for clst in 1:k    
        mu = med.(lwr[:,clst], V, upr[:,clst])
        all_mu[:,clst] = mu
        s_LB[clst] = norm(mu-V, 2)^2
    end
    best_LB, ~ = findmin(s_LB) # only need the best minimal distance, idx is not necessary
    return best_LB
end
    

# basic closed form
function getLowerBound_analytic_basic(X, k, lower=nothing, upper=nothing)
    d, n = size(X)
    LB = zeros(n)
    for s in 1:n
        # the way median is precalculated is faster 
        min_dist = getLB(X[:,s], k, d, lower, upper)
        LB[s] = min_dist
    end
    return maximum(LB)
end


# basic closed form
function getLowerBound_analytic_basic_FBBT(X, k, node, UB)
    d, n = size(X)
    lwr = node.lower
    upr = node.upper
    assign = node.assign
    firstcheck = seedInAllCluster(assign,k)  # If every cluster has seeds, firstcheck = true.
    addSeed = false

    LB_s_array = zeros(n)
    all_LB_s_array = zeros(k,n)
    for s in 1:n
        LB_s_array[s], all_LB_s, LB_clst_s = getLB_FBBT(view(X,:,s), k, d, lwr, upr, assign[s])
        if assign[s]== 0
            includ = all_LB_s .<= UB
            if sum(includ) == 1   ## exclude the memember of other clusters
                    c = Array(1:k)[includ][1]
                    assign[s] = c
                    addSeed = true
            else
                    all_LB_s_array[:,s] = all_LB_s         
            end
        end    
    end
    LB = max(maximum(LB_s_array), node.LB)
    
    for s in 1:n
        if assign[s] != -1
            UB_s, all_UB_s = getUB(view(X,:,s), k, d, lwr, upr, assign[s])
            if UB_s < LB
                assign[s] = -1
            end
            if assign[s] == 0
                all_LB_s = all_LB_s_array[:,s]
                ~, ind = findmin(all_LB_s)    
  
                all_LB_s[ind] = 1e16                
                if all_UB_s[ind] <= minimum(all_LB_s)
                    assign[s] = ind                   
                end

            end
        end
    end
    if addSeed
        secondcheck = seedInAllCluster(assign, k)
        if !firstcheck && secondcheck
            seeds_ind = haveSeedInAllCluster(assign, k)
            fbbt.expand_seeds(X, k, UB, seeds_ind)
            fbbt.updateassign(assign, seeds_ind, k)
            println("expand in LB")
        end
    end
    #println("#  Assigned:  ", NofAssignElements(assign))
    #println("#  Rm:   ", NofRmElements(assign))
    #println("#  remaining:   ", sum(assign.==0))
    return maximum(LB_s_array)
end




function NofAssignElements(assign)
    return length(assign) - sum(assign.==0) - sum(assign.==-1)
end

function NofRmElements(assign)
    return sum(assign.==-1)
end



function seedInAllCluster(assign, k)
    covered = falses(k)     
    for i in assign
        if i !=0 && i!=(-1)
            covered[i] = true
        end
    end
    return sum(covered) == k
end


function haveSeedInAllCluster(assign, k)
    seeds_ind = [[] for clst in 1:k]     
    for clst = 1:k
            s = indexin(clst, assign)[1]
            push!(seeds_ind[clst], s)
    end
    return seeds_ind 
end




function getLB_FBBT(V, k, d, lwr, upr, clst_assigned)
    if clst_assigned == -1 
       return 0, nothing, -1
    elseif clst_assigned == 0
        all_LB = zeros(k) # all distances between V and centers
        for clst in 1:k
            mu = med.(view(lwr, :, clst), V, view(upr, :, clst))
            all_LB[clst] = norm(mu-V, 2)^2
        end
        best_LB, ind, = findmin(all_LB) # only need the best minimal distance, idx is not necessary
        return best_LB, all_LB, ind
    else
        mu = med.(view(lwr, :, clst_assigned), V, view(upr, :, clst_assigned))
        return norm(mu-V, 2)^2, nothing, clst_assigned
    end
end




function getUB(V, k, d, lwr, upr, clst_assigned)
    if clst_assigned == -1
       return 0, nothing
    elseif clst_assigned == 0
        all_UB = zeros(k)       # all distances between V and centers
        for clst in 1:k
            mu = UB_sol.(view(lwr, :, clst), V, view(upr, :, clst))
            all_UB[clst] = norm(mu-V, 2)^2
        end
        best_UB, ~ = findmin(all_UB) # only need the best minimal distance, idx is not necessary
        return best_UB, all_UB
    else
        mu = UB_sol.(view(lwr, :, clst_assigned), V, view(upr, :, clst_assigned))
        return norm(mu-V, 2)^2, nothing
    end
end



function UB_sol(l,x,u)
    if abs(x-l) > abs(x-u)
       return l
    else
       return u
    end
end

# end of module
end