module kcenter_opt


using RDatasets, DataFrames, CSV
using Random, Distributions
using MLDataUtils, Clustering
using Statistics, LinearAlgebra
using Distances
using JuMP
using Ipopt, CPLEX#, SCIP

export global_OPT_L, global_OPT_x, global_OPT_L4_CPX

function obj_assign(centers, X)
    d, n = size(X)            
    dmat = pairwise(SqEuclidean(), X, centers, dims=2)
    costs = Vector{Float64}(undef, n)
    for j = 1:n
        costs[j] = minimum(view(dmat, j, :)) 
    end    
    return maximum(costs)
end


function init_bound(X, d, k, lower=nothing, upper=nothing)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d # get the feasible region of center
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
    return lower, upper
end

function max_dist(X, d, k, n, lower, upper)
    dmat_max = zeros(k,n)
    for j = 1:n
    	for i = 1:k
            max_distance = 0
            for t = 1:d
                max_distance += max((X[t,j]-lower[t,i])^2, (X[t,j]-upper[t,i])^2)
            end	
            dmat_max[i,j] += max_distance
	    end
    end    
    return dmat_max
end

function sel_closest_centers(centers, X)
    d,k = size(centers)
    t_ctr = zeros(d,k)
    dmat = pairwise(SqEuclidean(), X, centers, dims=2)
    for j in 1:k
        c, a = findmin(dmat[:,j])
        t_ctr[:,j] = X[:,a]
    end
    return t_ctr
end

function global_OPT_x(X, k, thread_flag, lower=nothing, upper=nothing, mute = false)  # l2 norm 
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)
    inter_max = upper-lower


    m = direct_model(CPLEX.Optimizer())
    if thread_flag == false # single thread
        MOI.set(m, MOI.NumberOfThreads(), 1)
    end
    set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 3600*4) # maximum runtime limit is time_lapse*16 or set to 4/12 hours
    # set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 1) # 0 for qcp relax and 1 for lp oa relax.
    # set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand())
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand())
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ))
    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1)
    @variable(m, costs[j in 1:n], start=rand())
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= 0)
    @variable(m, lambda[1:k, 1:n], Bin)
    @constraint(m, [i in 1:k], sum(lambda[i,j] for j in 1:n) == 1)
    @constraint(m, [i in 1:k, j in 1:n], b[i,j] >= lambda[i,j] ) 
    @constraint(m, [i in 1:k, j in 1:n], X[:,j]- centers[:,i] .>= -inter_max[:,i]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], X[:,j] - centers[:,i] .<= inter_max[:,i]*(1-lambda[i,j]))
    @variable(m, cost>=0, start =rand())
    @constraint(m, [j in 1:n], cost>=costs[j])

    @objective(m, Min, cost)

    optimize!(m)
    cplex_centers = value.(centers)
    cplex_lb = min(objective_bound(m), objective_value(m))
    cplex_ub = max(objective_bound(m), objective_value(m))
    gap = relative_gap(m)*100 # get the relative gap for cplex solver
    centers = sel_closest_centers(cplex_centers, X)
    objv = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    println("cplex_objective_LB: ", cplex_lb)
    println("cplex_objective_UB: ", cplex_ub)
    println("cplex_gap: ", gap)
    println("real_UB: ", objv)
    println("real_gap: ", (objv - cplex_lb)/objv*100)
    println("node number:", JuMP.node_count(m))
    return centers, objv, (objv - cplex_lb)/objv*100, cplex_lb, JuMP.node_count(m)
end


function global_OPT_L(X, k, nlines, thread_flag, lower=nothing, upper=nothing, mute = false)  # l2 norm 
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    println("lower: ", lower)
    println("upper: ", upper)
    # lu_mean = (lower + upper) / 2.0
    dmat_max = max_dist(X, d, k, n, lower, upper)
    inter_max = upper-lower
    # control line numbers
    # nlines = Int(round(d*n / 128)) 
    println("2*nlines: ", 2*nlines)

    m = direct_model(CPLEX.Optimizer())
    if thread_flag == false # single thread
        MOI.set(m, MOI.NumberOfThreads(), 1)
    end
    set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 3600*4) # maximum runtime limit is time_lapse*16 or set to 4/12 hours
    # set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 1) # 0 for qcp relax and 1 for lp oa relax.
    # set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand())
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand())

    ##### Piecewise Linear Constraints #####
    @variable(m, 0 <= w[t in 1:d, i in 1:k], start=rand()) # add the horizontal line of the lower bottom line bound 
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
    itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
    for line in 0:(nlines-1)
        lwr = lower+itval.*line
        upr = upper-itval.*line
        @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
        @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
    end

    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1)
    # @constraint(m, [i in 1:k, j in 1:n], b[i,j] <= dmat[i,j])
    @variable(m, costs[j in 1:n], start=rand())
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= 0)
    @variable(m, lambda[1:k, 1:n], Bin)
    @constraint(m, [i in 1:k], sum(lambda[i,j] for j in 1:n) == 1)
    @constraint(m, [i in 1:k, j in 1:n], b[i,j] >= lambda[i,j] ) 
    @constraint(m, [i in 1:k, j in 1:n], X[:,j]- centers[:,i] .>= -inter_max[:,i]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], X[:,j] - centers[:,i] .<= inter_max[:,i]*(1-lambda[i,j]))
    @variable(m, cost>=0, start =rand())
    @constraint(m, [j in 1:n], cost>=costs[j])

    @objective(m, Min, cost)

    optimize!(m)


    cplex_centers = value.(centers)
    cplex_lb = min(objective_bound(m), objective_value(m))
    cplex_ub = max(objective_bound(m), objective_value(m))
    gap = relative_gap(m)*100 # get the relative gap for cplex solver
    println("cplex_centers: ", cplex_centers)
    centers = sel_closest_centers(cplex_centers, X)
    println("centers: ", centers)
    objv = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    println("cplex_objective_LB: ", cplex_lb)
    println("cplex_objective_UB: ", cplex_ub)
    println("cplex_gap: ", gap)
    println("real_UB: ", objv)
    println("real_gap: ", (objv - cplex_lb)/objv*100)
    println("node number:", JuMP.node_count(m))
    return centers, objv, (objv - cplex_lb)/objv*100, cplex_lb, JuMP.node_count(m)
end

function global_OPT_L4_CPX(X, k, nlines, thread_flag, lower=nothing, upper=nothing, mute = false)  # l2 norm 
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    println("lower: ", lower)
    println("upper: ", upper)
    # lu_mean = (lower + upper) / 2.0
    dmat_max = max_dist(X, d, k, n, lower, upper)
    inter_max = upper-lower
    # control line numbers
    # nlines = Int(round(d*n / 128)) 
    println("2*nlines: ", 2*nlines)

    m = direct_model(CPLEX.Optimizer())
    if thread_flag == false # single thread
        MOI.set(m, MOI.NumberOfThreads(), 1)
    end
    set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 1)
    # set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_Search", 1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 3600*4) # maximum runtime limit is time_lapse*16 or set to 4/12 hours
    # set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 1) # 0 for qcp relax and 1 for lp oa relax.
    # set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    @variable(m, lambda[1:k, 1:n], Bin) # k*n-1
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand())
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand())
    ##### Piecewise Linear Constraints #####
    @variable(m, 0 <= w[t in 1:d, i in 1:k], start=rand()) # add the horizontal line of the lower bottom line bound 
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
    itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
    for line in 0:(nlines-1)
        lwr = lower+itval.*line
        upr = upper-itval.*line
        @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
        @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
    end

    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1)
    # @constraint(m, [i in 1:k, j in 1:n], b[i,j] <= dmat[i,j])
    @variable(m, costs[j in 1:n], start=rand())
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= 0)
   
    @constraint(m, [i in 1:k], sum(lambda[i,j] for j in 1:n) == 1)
    @constraint(m, [i in 1:k, j in 1:n], b[i,j] >= lambda[i,j] ) 
    @constraint(m, [i in 1:k, j in 1:n], X[:,j]- centers[:,i] .>= -inter_max[:,i]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], X[:,j] - centers[:,i] .<= inter_max[:,i]*(1-lambda[i,j]))
    @variable(m, cost>=0, start =rand())
    @constraint(m, [j in 1:n], cost>=costs[j])

    @objective(m, Min, cost)
    ###### Set callbcak function ######
    call_num = 0
    function my_callback_function(cb_context::CPLEX.CallbackContext, context_id::Clong)
        if context_id == CPX_CALLBACKCONTEXT_RELAXATION
        # println("call_num: ", call_num)
            call_num += 1
            lambda_lb = Array{Ref{Cdouble}}(zeros(k, n).+2)
            lambda_ub = Array{Ref{Cdouble}}(zeros(k, n).+2)
            j_lambda_lb = zeros(k, n)
            j_lambda_ub = zeros(k, n)
            for i in 1:k
                for j in 1:n
                    CPXcallbackgetlocallb(cb_context, lambda_lb[i, j], Cint((j-1)*k+i-1), Cint((j-1)*k+i-1))
                    CPXcallbackgetlocalub(cb_context, lambda_ub[i, j], Cint((j-1)*k+i-1), Cint((j-1)*k+i-1))
                    j_lambda_lb[i, j] = lambda_lb[i, j][]
                    j_lambda_ub[i, j] = lambda_ub[i, j][]
                end
            end
            ############## make user cut ################
            # println("############### current node info - $call_num ################")
            # println("centers_lb: ", centers_lb)
            # println("centers_ub: ", centers_ub)
            # println("############### make user cut ################")
            for i in 1:k # centers[t in 1:d, i in 1:k]
                idxs = findall(x -> x==1.0, j_lambda_ub[i, :])
                if length(idxs) > 0 && length(idxs) != n
                    tmp_centers_lb = minimum(X[:, idxs], dims = 2)
                    tmp_centers_ub = maximum(X[:, idxs], dims = 2)
                    # println(tmp_centers_lb)
                    # println(tmp_centers_ub)
                    # Ref{Cint}(k*n-1+(k-1)*d+t)
                    for t in 1:d
                        # ########## Set the lower bound ##########
                        rmatind = [Cint(k*n-1+(i-1)*d+t)] # Ref(Cint(k*n-1+(i-1)*d+t)), Ref(Cdouble(1.0))
                        rmatval = [1.0]
                        status = CPXcallbackaddusercuts(cb_context, Cint(1), Cint(1), Ref(Cdouble(tmp_centers_lb[t])), Ref(Cchar('G')), Ref{Cint}(0), pointer(rmatind), pointer(rmatval), Ref{Cint}(CPX_USECUT_FORCE), Ref{Cint}(1))
                        # # println(status)
                        # # ########## Set the upper bound ##########
                        status = CPXcallbackaddusercuts(cb_context, Cint(1), Cint(1), Ref(Cdouble(tmp_centers_ub[t])), Ref(Cchar('L')), Ref{Cint}(0), pointer(rmatind), pointer(rmatval), Ref{Cint}(CPX_USECUT_FORCE), Ref{Cint}(1))
                    end
                    # println("submit new constraints for centers - $i")
                end
            end            
       end # for if
    end # for function
    MOI.set(m, CPLEX.CallbackFunction(), my_callback_function)

    optimize!(m)

    cplex_centers = value.(centers)
    cplex_lb = min(objective_bound(m), objective_value(m))
    cplex_ub = max(objective_bound(m), objective_value(m))
    gap = relative_gap(m)*100 # get the relative gap for cplex solver
    println("cplex_centers: ", cplex_centers)
    centers = sel_closest_centers(cplex_centers, X)
    println("centers: ", centers)
    objv = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    println("cplex_objective_LB: ", cplex_lb)
    println("cplex_objective_UB: ", cplex_ub)
    println("cplex_gap: ", gap)
    println("real_UB: ", objv)
    println("real_gap: ", (objv - cplex_lb)/objv*100)
    println("node number:", JuMP.node_count(m))
    return centers, objv, (objv - cplex_lb)/objv*100, cplex_lb, JuMP.node_count(m)
end


# end of module
end

    # ###### Set callbcak function ######
    # call_num = 0
    # function my_callback_function(cb_context::CPLEX.CallbackContext, context_id::Clong)
    #     if context_id == CPX_CALLBACKCONTEXT_BRANCHING 
    #         call_num += 1
    #         println("call_num: ", call_num)
    #         centers_lb = Array{Ref{Cdouble}}(zeros(d, k))
    #         centers_ub = Array{Ref{Cdouble}}(zeros(d, k))
    #         j_centers_lb = zeros(d, k)
    #         j_centers_ub = zeros(d, k)
    #         for t in 1:d
    #             for i in 1:k # centers[t in 1:d, i in 1:k] (i-1)*d+t-1
    #                 CPXcallbackgetlocallb(cb_context, centers_lb[t, i], Cint((i-1)*d+t-1), Cint((i-1)*d+t-1))
    #                 CPXcallbackgetlocalub(cb_context, centers_ub[t, i], Cint((i-1)*d+t-1), Cint((i-1)*d+t-1))
    #                 j_centers_lb[t, i] = centers_lb[t, i][]
    #                 j_centers_ub[t, i] = centers_ub[t, i][]
    #             end
    #         end
    #         if call_num % 50 == 0
    #             println(j_centers_lb)
    #             println(j_centers_ub)
    #         end

    #         for i in 1:k
    #             for j in 1:n
    #                 for t in 1:d
    #                     if X[t, j] < j_centers_lb[t, i]
    #                         CPXcallbackchgbds(cb_context, Cint((i-1)*d+t-1), Cint(j-1), Cint(1), Cdouble(j_centers_lb[t, i]))
    #                         # con_lb = @build_constraint(lambda[i, j] * X[t, j] >= j_centers_lb[t, i] for i in 1:k, j in 1:n, t in 1:d)
    #                     end
    #                     if j_centers_ub[t, i] < centers[t, i]
    #                         CPXcallbackchgbds(cb_context, Cint((i-1)*d+t-1), Cint(j-1), Cint(1), Cdouble(j_centers_ub[t, i]))
    #                         # con_lb = @build_constraint(lambda[i, j] * X[t, j] >= j_centers_lb[t, i] for i in 1:k, j in 1:n, t in 1:d)
    #                     end
    #                 end
    #             end
    #         end
    #         # con_lb = @build_constraint(lambda[i, j] * X[t, j] >= j_centers_lb[t, i] for i in 1:k, j in 1:n, t in 1:d)
    #         # con_ub = @build_constraint(lambda[i, j] * X[t, j] <= j_centers_ub[t, i] for i in 1:k, j in 1:n, t in 1:d)
    #         # MOI.submit(model, MOI.LazyConstraint(cb_context), con_lb)
    #         # MOI.submit(model, MOI.LazyConstraint(cb_context), con_ub)
    #         println("submit")
    #         # vars = Ref{CPXINT}(0)
    #         # CPXgetcallbacknodeinfo(backend(m).env, cb_context, context_id, Cint(0), CPX_CALLBACK_INFO_NODE_VAR, vars)
    #         # println("vars: ", vars[])
    #         # tmp = Array{Ref{Cdouble}}(zeros(4))
    #         # tmp2 = Array{Ref{Cdouble}}(zeros(4))
    #         # for i = 1:4
    #         #     CPXcallbackgetlocallb(cb_context, tmp[i], Cint(i-1), Cint(i-1))
    #         #     CPXcallbackgetlocalub(cb_context, tmp2[i], Cint(i-1), Cint(i-1))
    #         #     if i == 4
    #         #         CPXcallbackgetlocallb(cb_context, tmp[i], Cint(vars[]), Cint(vars[]))
    #         #         CPXcallbackgetlocalub(cb_context, tmp2[i], Cint(vars[]), Cint(vars[]))
    #         #     end
    #         #     print("$i-lb: ", tmp[i][], ", ub: ", tmp2[i][], "; ")
    #         # end
    #         # println()
    #     end
    #     # if tmp2[2][] > 0
    #     #     con = @build_constraint(x <= tmp2[2][]) # x < ub[y]
    #     #     MOI.submit(model, MOI.UserCut(cb_context), con)
    #     #     println("New con: ", con)
    #     # end

    #     # CPLEX.cbcandidateispoint(cb_context) == 0 && return
    #     # vars = Ref{CPXINT}()
    #     # CPXgetcallbacknodeinfo(backend(m).env, cb_context, context_id, Cint(0), CPX_CALLBACK_INFO_NODE_VAR, vars)
    #     # println("vars: ", vars)
    #     # tmp = Ref{Cdouble}()
    #     # CPXcallbackgetlocallb(cb_context, tmp, Cint(0), Cint(0))
    #     # println("lb:", tmp)
    #     # tmp2 = Ref{Cdouble}()
    #     # CPXcallbackgetlocalub(cb_context, tmp2, Cint(0), Cint(0))
    #     # println("ub: ", tmp2)
    #     # MOI.get(m, Cplex.BranchType.BranchOnVariable)
        
    #     # c_b = m[:b]
    #     # c_lambda = m[:lambda]
    #     # c_dmat = m[:dmat]
    #     # c_centers = m[:centers]

    #     # k, n = size(b)

    #     # b_count = 0
    #     # lambda_count = 0
    #     # for i in 1:k
    #     #     for j in 1:n
    #     #         if has_lower_bound(c_b[i, j]) && has_upper_bound(c_b[i, j])
    #     #             print("has_lower_bound(c_b[", i, ", ", j, "]) ", has_lower_bound(c_b[i, j]))
    #     #             println(", has_upper_bound(c_b[", i, ", ", j, "]) ", has_upper_bound(c_b[i, j]))
    #     #             if lower_bound(c_b[i, j]) == upper_bound(c_b[i, j])
    #     #                 b_count += 1
    #     #             end
    #     #         end
    #     #         if has_lower_bound(c_lambda[i, j]) && has_upper_bound(c_lambda[i, j])
    #     #             print("has_lower_bound(c_lambda[", i, ", ", j, "]) ", has_lower_bound(c_lambda[i, j]))
    #     #             println(", has_upper_bound(c_lambda[", i, ", ", j, "]) ", has_upper_bound(c_lambda[i, j]))
    #     #             if lower_bound(c_lambda[i, j]) == upper_bound(c_lambda[i, j])
    #     #                 lambda_count += 1
    #     #             end
    #     #         end
    #     #     end
    #     # end
    #     # println("b_count: ", b_count)
    #     # println("lambda_count: ", lambda_count)

    #     # for i in 1:d
    #     #     for j in 1:k
    #     #         if has_lower_bound(c_centers[i, j]) && has_upper_bound(c_centers[i, j])
    #     #             print(", l[", i, ", ", j, "] ", lower_bound(c_centers[i, j]))
    #     #             print(", u[", i, ", ", j, "] ", upper_bound(c_centers[i, j]))
    #     #         end
    #     #     end
    #     # end
    #     # status = callback_node_status(cb_data, m)
    #     # if status == MOI.CALLBACK_NODE_STATUS_FRACTIONAL
    #     #     println(" - Solution is integer infeasible!")
    #     # elseif status == MOI.CALLBACK_NODE_STATUS_INTEGER
    #     #     println(" - Solution is integer feasible!")
    #     # else
    #     #     @assert status == MOI.CALLBACK_NODE_STATUS_UNKNOWN
    #     #     println(" - I don't know if the solution is integer feasible :(")
    #     # end
    # end
    # MOI.set(m, CPLEX.CallbackFunction(), my_callback_function)
    # # MOI.set(m, MOI.UserCutCallback(), my_callback_function)

# end of module
