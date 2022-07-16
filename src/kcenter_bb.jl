module kcenter_bb

using RDatasets, Clustering
using Printf
using Random, Statistics
using LinearAlgebra
using Distances
using kcenter_ub, kcenter_lb, kcenter_opt, branch, Nodes, fbbt
export branch_bound

maxiter = 5000000
tol = 1e-6
mingap = 1e-3
time_lapse = 4*3600 # 4 hours


# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())


# during iteration: 
# LB = node.LB is the best LB among all node and is in current iteration
# UB is the best UB, and node_UB the updated UB(it is possible: node_UB > UB) of current iteration (after run getUpperBound)
# node_LB is the updated LB of current itattion (after run probing or getLowerBound_adptGp_LD)

function branch_bound(X, k, bt_method = "FBBT", symmtrc_breaking = 1)
    
    #Scale the data
    x_max = maximum(X) # max value after transfer to non-zero value
    tnsf_max = false
    if x_max > 20
        tnsf_max = true
        X = X/(x_max*0.05)
    end
        
    d, n = size(X);
    lower, upper = kcenter_opt.init_bound(X, d, k)
    max_LB = 1e15; # used to save the best lower bound at the end (smallest but within the mingap)
    centers, UB = kcenter_ub.getUpperBound(X, k, lower, upper, tol)
    root = Node(lower, upper, -1, -1e15, zeros(Int, n), trues(n, k));

    if bt_method == "FBBT"
        assign = zeros(Int, n)
        findseeds  = fbbt.select_seeds(X, k, UB, root.assign, 10)
        if !findseeds
            symmtrc_breaking = 1
        else
            oldboxSize = sum(upper-lower)   #norm(upper-lower, 2)^2/k
            println("sum:   ", oldboxSize)
            oldUB = UB
            root_LB = -1e15
            stuck = 0
            for t=1:40
                println("trial      ",t,    "   fbbt ")
                root.lower, root.upper = fbbt.fbbt_base(X, k, root, UB, 50)
                boxSize = sum(root.upper-root.lower)  #norm(root.upper-root.lower, 2)^2/k 
                println("sum:   ", boxSize)
                centers, UB = randomUB(X, root.lower, root.upper, UB, centers, 10)
                #println("UB  ", UB)
                root_LB = kcenter_lb.getLowerBound_analytic_basic_FBBT(X, k, root, UB)
                root.LB = root_LB
                
                if boxSize/oldboxSize >= 0.99  && UB/oldUB >= 0.999
                    stuck += 1
                else
                    stuck = 0
                end
                if stuck == 2    
                    break
                end
                oldboxSize = boxSize
                oldUB = UB

                if (UB-root_LB) <= mingap*min(abs(root_LB), abs(UB))
                    println("LB   ", root_LB, "  UB  ", UB, " Gap ", (UB-root_LB)/min(abs(root_LB), abs(UB)))
 
                    ctr_dist = pairwise(SqEuclidean(),centers,centers)
                    min_ctr_dist = ctr_dist[1,2]
                    for i in 1:k
                        for j in 1:k
                            if i != j && ctr_dist[i,j] < min_ctr_dist
                                min_ctr_dist = ctr_dist[i,j]
                            end
                        end
                    end
                    rate = min_ctr_dist/UB
                    println("rate = minimum distance between centers/UB =   ", rate)
                
                    # transfer back to original value of optimal value
                    if tnsf_max
                        UB = UB .* (x_max*0.05)^2
                    end    
                    
                    println("UB:  ", UB)
                    
                    return centers, UB, nothing
                end

                remain = can_center_or_assign(root.assign, root.center_cand)
                println("# of elements deleted", n - sum(remain))
                println("# of elements remain", sum(remain))
                if sum(remain)/n <= 0.8
                    X=X[:, remain]
                    root.assign = root.assign[remain]
                    root.center_cand = root.center_cand[remain, :]
                    n = sum(remain)
                end
            end
        end
    end    

    
    # groups is not initalized, will generate at the first iteration after the calculation of upper bound
    nodeList =[root]
    iter = 0
    println(" iter ", " left ", " lev  ", "       LB       ", "       UB      ", "      gap   ")

    # get program end time point
    end_time = time_finish(time_lapse) # the branch and bound process ends after 6 hours

    #####inside main loop##################################
    calcInfo = [] # initial space to save calcuation information
    while nodeList!=[]
        # we start at the branch(node) with lowest Lower bound
	    LB, nodeid = kcenter_lb.getGlobalLowerBound(nodeList) # Here the LB is the best LB and also node.LB of current iteration
        node = nodeList[nodeid]
        deleteat!(nodeList, nodeid)
        
        # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
        if iter%10 == 0
            @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        end
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])
        
        # time stamp should be checked after the retrival of the results
        if (iter == maxiter) || (time_ns() >= end_time)
            break
        end        

        ############# iteratively bound tightening #######################
        iter += 1   
        
        if bt_method == "FBBT"           
            #println("FBBT   ")
            node.lower, node.upper = fbbt.fbbt_base(X, k, node, UB) 
        else
            node.lower, node.upper = kcenter_lb.sel_set(X, k, node.lower, node.upper) 
        end    
        if node.lower == nothing && node.upper == nothing
            continue
        #else
            #println("boxsize :   ",sum(node.upper-node.lower))  #norm(node.upper-node.lower, 2)^2/k)
        end

        node_LB = LB
        ##### LB ###############
        #println("LB:  ")
        # getLowerBound with closed-form expression
        if (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            println("analytic LB  ",node_LB, "   >=UB    ", UB)
        else 
            if bt_method == "FBBT"
                node_LB = kcenter_lb.getLowerBound_analytic_basic_FBBT(X, k, node, UB) 
            else
                node_LB = kcenter_lb.getLowerBound_analytic_basic(X, k, node.lower, node.upper) 
            end
        end
        #node_LB = max(node_LB, LB)
        ##### UB ####################################
        ##### get upper bound from random centers
        #println("UB:  ")
        oldUB = UB
        centers, UB = randomUB(X, node.lower, node.upper, UB, centers)
        if (UB < oldUB)
            # the following code delete branch with lb close to the global upper bound
            delete_nodes = []
            for (idx,nd) in enumerate(nodeList)
                if (UB-nd.LB) <=mingap*min(abs(UB), abs(nd.LB))
                    push!(delete_nodes, idx)
                end
            end
            deleteat!(nodeList, sort(delete_nodes))
        end

        
        if iter%50 == 0 && bt_method == "FBBT"
            remain = nothing
            println("root node reduce size")
            if length(nodeList) >= 500
                root.lower, root.upper = getUnionBound(nodeList)
                for t=1:2
                    #println("trial  ",t, "  fbbt ")
                    root.lower, root.upper = fbbt.fbbt_base(X, k, root, UB, 50)
                    boxSize = sum(root.upper-root.lower)   #norm(root.upper-root.lower, 2)^2/k
                    println("sum of root:   ", boxSize)
                    root.LB = max(root.LB, LB)
                    root_LB = kcenter_lb.getLowerBound_analytic_basic_FBBT(X, k, root, UB)
                    #println("root LB:      ", root.LB)
                    root.LB = max(root.LB, root_LB)

                    if (UB-root_LB) <= mingap*min(abs(root_LB), abs(UB))
                        println("LB   ", root_LB, "  UB  ", UB)
                        # transfer back to original value of optimal value
                        if tnsf_max
                            UB = UB .* (x_max*0.05)^2
                        end    
                    
                        println("UB:  ", UB)

                        return centers, UB, nothing
                    end
                    remain = can_center_or_assign(root.assign, root.center_cand)
                    #println("# of elements deleted", n - sum(remain))
                    #println("# of elements remain", sum(remain))
                end
            else                    
                for (idx,nd) in enumerate(nodeList)
                    node_remain = can_center_or_assign(nd.assign, nd.center_cand)                 
                    if idx == 1
                        remain = node_remain                
                    else
                        remain = (remain .| node_remain)    
                    end
                    node_remain = can_center_or_assign(node.assign, node.center_cand)                 
                    remain = (remain .| node_remain)    
                    #println("node:  ", idx,  "   LB    ", nd.LB, "  level  ", nd.level, "    # of elements deleted  ", n - sum(node_remain))
                    #println("# of elements deleted", n - sum(remain))
                end
            end     
            if sum(remain)/n <= 0.8
                    println(n - sum(remain), " of elements deleted")
                    println(sum(remain), " of elements remain")         
                    n = sum(remain)
                    X=X[:, remain]
                    root.assign = root.assign[remain]
                    root.center_cand = root.center_cand[remain, :]
                    for (idx,nd) in enumerate(nodeList)
                        nd.assign = nd.assign[remain]
                        nd.center_cand = nd.center_cand[remain, :]
                    end
                    node.assign = node.assign[remain]
                    node.center_cand = node.center_cand[remain, :]
            end
        end

        
        # here this condition include the condition UB < node_LB and the condition that current node's LB is close to UB within the mingap
        # Such node no need to branch
        if (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            # save the best LB if it close to UB enough (within the mingap)
            if node_LB < max_LB
                max_LB = node_LB
            end
        else
            bVarIdx, bVarIdy = branch.SelectVarMaxRange(node)
            #println("branching on ", bVarIdx,"    ", bVarIdy )
            # the split value is chosen by the midpoint
            bValue = (node.upper[bVarIdx,bVarIdy] + node.lower[bVarIdx,bVarIdy])/2;
            branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k, bt_method, symmtrc_breaking);
        end
    end

    if nodeList==[]
        println("all node solved")
        # save final calcuation information
        push!(calcInfo, [iter, length(nodeList), max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
    else
        max_LB = calcInfo[end][4]
    end
    println("solved nodes:  ",iter)
    
    @printf "%-52d  %-14.4e %-14.4e %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"
    println("centers   ",centers)
    
    ctr_dist = pairwise(SqEuclidean(),centers,centers)
    min_ctr_dist = ctr_dist[1,2]
    for i in 1:k
        for j in 1:k
            if i != j && ctr_dist[i,j] < min_ctr_dist
                min_ctr_dist = ctr_dist[i,j]
            end
        end
    end
    rate = min_ctr_dist/UB
    println("rate = minimum distance between centers/UB =   ", rate)
    

    # transfer back to original value of optimal value
    if tnsf_max
        UB = UB .* (x_max*0.05)^2
    end    
    
    println("UB:  ", UB)

    # UB = UB .* (x_max-x_min)^2
    # println("Initial UB:  ", UB)
    
    return centers, UB, calcInfo
end






function getUnionBound(nodeList)
    lower = nothing
    upper = nothing    
    for (idx, nd) in enumerate(nodeList)
        if idx == 1
            lower = copy(nd.lower)
            upper = copy(nd.upper)
        else
            lower = min.(lower, nd.lower)
            upper = max.(upper, nd.upper)
        end
    end
    return lower, upper
end


function randomUB(X, lwr, upr, UB, centers, ntr=5)
    ctr = centers     
    d, k = size(lwr)     
    t_centers = (lwr.+upr)/2
    for tr = 1:ntr
        t_ctr = kcenter_opt.sel_closest_centers(t_centers, X)
        t_UB = kcenter_opt.obj_assign(t_ctr, X)
        if (t_UB < UB)
            UB = t_UB
            ctr = t_ctr
        end
        if tr !=ntr
            t_centers = rand(d, k).*(upr - lwr) .+ lwr
        end
    end
    return ctr, UB
end


function can_center_or_assign(assign, center_cand)
    n = length(assign)     
    remain = trues(n)
    for s in 1:n
        if assign[s] ==  -1
            if sum(center_cand[s,:]) == 0
                remain[s] = false
            end
        end
    end
    return remain
end
 





# end of module
end
