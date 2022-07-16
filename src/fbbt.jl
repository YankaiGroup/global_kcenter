module fbbt
using Distances, LinearAlgebra, StatsBase
using kcenter_ub


function select_init_seeds(X, k, UB)   # randomly select the start point.
    d,n = size(X)
    init_seeds = zeros(d,k)
    init_seeds_ind = zeros(Int, k)
    z = rand(1:n,1)[1]  # randomly select a sample as the beginning
    init_seeds[:,1] = X[:,z]
    init_seeds_ind[1] = z  
  
    diffclst = 1
    for j = 2:k
        ~, a = kcenter_ub.max_dist(X, init_seeds[:,1:(j-1)])
        x = X[:,a]
        find_ = true
        for i in 1:(j-1)
            if norm(x - init_seeds[:,i],2)^2 < 4*UB
                find_ = false
                break
            end
        end
        if !find_
            diffclst = 0
            break
        else
            init_seeds[:,j] = x
            init_seeds_ind[j] = a
        end
    end
    return diffclst, init_seeds_ind
end


function select_seeds(X, k, UB, assign, n_trial=1)
    println("UB:   ", UB)
    d,n = size(X)
    iter = 0
    diffclst = 0
    init_seeds_ind = []
    while iter <= 100 && diffclst == 0   # try 100 times until find the real seeds, then end the loop. Or 100 times still can not find the real seeds, then end the loop.
        diffclst, init_seeds_ind = select_init_seeds(X,k, UB)
        iter +=1
    end
    println("try", iter,"times to select seeds")
    findseed = false
    if diffclst == 1
        println("Find seeds successfully.")
        findseed = true
    else
        println("Can not find seeds after 100 times.")
    end

    seeds_ind = []
    for clst = 1:k
        push!(seeds_ind, [])
    end
    
    if diffclst == 1   # check if find the real seeds
        for clst = 1:k
            push!(seeds_ind[clst], init_seeds_ind[clst])
        end
        expand_seeds(X, k, UB, seeds_ind, n_trial)
        fbbt.updateassign(assign, seeds_ind, k)
    end    
    return findseed #, seeds_ind
end


function alreadyin(s, seeds_ind, k)
    exist = false
    for clst = 1:k
        if  in(s, seeds_ind[clst])
            exist = true
            break
        end
    end    
    return exist
end


function updateassign(assign, seeds_ind, k)
    n = length(assign)
    for clst = 1:k
        for i = 1:length(seeds_ind[clst])
            ind = seeds_ind[clst][i]
            if assign[ind] != -1
                assign[ind] = clst
            end
        end
    end
end

function seedInAllCluster(seeds_ind)
    all = true
    for clst = 1:length(seeds_ind)
        if length(seeds_ind[clst]) == 0
            all = false
            break
        end
    end
    return all
end

function expand_seeds(X, k, UB, seeds_ind, n_trial=1)
    d,n = size(X)
    check = seedInAllCluster(seeds_ind)
    if !check
        println("error! need to have at least one seed per cluster first!")
    end

    inner_assign = zeros(Int, n)
    for clst = 1:k
        for ind in seeds_ind[clst]
            inner_assign[ind] = clst
        end
    end

    init_seeds = zeros(d,k)
    for clst = 1:k
        init_seeds[:,clst] = X[:, seeds_ind[clst][1]]
    end

    for t = 1:n_trial    
        for s = 1:n
            if inner_assign[s] == 0
                includ = falses(k)
                x = X[:,s]
                for clst = 1:k
                    d = norm(x - view(init_seeds,:, clst) , 2)^2
                    if d < 4*UB
                        includ[clst] = true
                    end
                end
                if sum(includ) == 1   ## exclude the memember of other clusters
                    c = Array(1:k)[includ][1]
                    push!(seeds_ind[c], s)
                    inner_assign[s] = c
                end
            end
        end

        for clst = 1:k
            ix = rand(1:length(seeds_ind[clst]))
            init_seeds[:,clst] = X[:, seeds_ind[clst][ix]]
        end
    end

    num = 0
    for clst = 1:length(seeds_ind)
        num += length(seeds_ind[clst])
    end
    println("Expand #  seeds ",   num)
    return seeds_ind
end





function divideclusternorm(X, UB, k, lower, upper, assign, center_cand, max_nseeds_c = 20)      #given the seeds selected. Ddivide dataset according to norm <= UB
    d, n =size(X)
    lwr = copy(lower)
    upr = copy(upper)
    for clst = 1:k
        old_center_cand_clst = center_cand[:, clst]
        oldset = Array(1:n)[old_center_cand_clst]
        seeds_ind_c =  findall(x->x==clst, assign)   #seeds_ind[clst]

        if length(seeds_ind_c) <= max_nseeds_c
            nseed_c = length(seeds_ind_c)
            seeds_id = seeds_ind_c[1:nseed_c]
            seeds =  X[:, seeds_id]
        else         
            seeds = kcenter_ub.fft_FBBT(view(X, :, seeds_ind_c), max_nseeds_c, lower[:,clst], upper[:,clst])
            #=
            if length(seeds_ind_c)>= 100000
                inds = seeds_ind_c[sample(1:length(seeds_ind_c), 100000, replace=false)]
                seeds = kcenter_ub.fft(view(X, :, inds), max_nseeds_c, lower[:,clst], upper[:,clst])
            else
                seeds = kcenter_ub.fft(view(X, :, seeds_ind_c), max_nseeds_c, lower[:,clst], upper[:,clst])
            end
            =#             
        end    
        dmat = pairwise(SqEuclidean(),  seeds, X[:, oldset])

        #=
        nseed_c = min(length(seeds_ind_c), max_nseeds_c)
        seeds_id = seeds_ind_c[1:nseed_c]  ### To do select seeds far away from each other
        dmat = pairwise(SqEuclidean(),  X[:, seeds_id], X[:, oldset])        
        =#
        num = 0
        center_cand[:, clst] .= false
        for j in 1:length(oldset)
            s = oldset[j]
            if sum(view(X,:,s) .>= view(lower,:,clst))==d && sum(view(X,:,s) .<= view(upper,:,clst))==d
                if sum( view(dmat,:,j) .> UB) == 0
                    center_cand[s, clst] = true    
                    num += 1
                    if num == 1
                        lwr[:, clst]=X[:, s]
                        upr[:, clst]=X[:, s]
                    else
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
        end
        
           #println(num, "# center candidate left in cluster  ", clst)
        if num == 0
            return nothing, nothing
        elseif num == 1    
            #println("There exists some cluster which only contains one sample. Use this sample as new seeds.")   # check if the max of distance greater than UB
            s = Array(1:n)[center_cand[:, clst]][1]
            if assign[s] == 0
                assign[s] = clst
            end
        end
    end
    #println("#elements cannot be centers  ", n-sum(maximum(center_cand, dims=2)))
    return lwr, upr
end




function fbbt_base(X, k, node, UB, max_nseeds_c = 20)
    d, n = size(X)
    assign = node.assign
    center_cand = node.center_cand

    lwr = copy(node.lower)
    upr = copy(node.upper)
    ra = sqrt(UB)

    for i = 1:n
        if assign[i] != 0 && assign[i] != -1
            clst = assign[i]
            for j in 1:d
                if lwr[j,clst] < X[j,i] - ra
                    lwr[j,clst] = X[j,i] - ra
                end
                if upr[j,clst] > X[j,i] + ra
                    upr[j,clst] = X[j,i] + ra
                end
            end
        end
    end
    for clst = 1:k
            if sum(view(lwr,:,clst) .<=view(upr,:,clst)) != d
                println("Delete this node")  # intersection is empty, delete this node
                return nothing, nothing
            end
    end   

    #println("divideclusternorm:    ")    
    lwr, upr = divideclusternorm(X, UB, k, lwr, upr, assign, center_cand, max_nseeds_c)  # given three seeds, divide the dataset according to distance<=UB
    if lwr == nothing && upr == nothing
        println("Delete this node")  # intersection is empty, delete this node
        return nothing, nothing
    end
    return lwr, upr
end






# end of module
end
