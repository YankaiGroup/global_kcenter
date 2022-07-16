module kcenter_ub

using Clustering, Distances, LinearAlgebra
using kcenter_opt


# fft
function max_dist(X1, X2)  # max distance from array X1 to array X2
    d,n = size(X1)
    dist_matrix = pairwise(SqEuclidean(),X1,X2, dims=2)
    dist = zeros(n)
    for i = 1:n
        dist[i] = minimum(view(dist_matrix,i,:))
    end
    c, a = findmax(dist)
    return c, a  # c is the max distance, a is the index.
end

#=
function fft(X,k)
    d,n = size(X)
    centers = zeros(d,k)
    z0 = rand(1:n,1)  # randomly select a sample as the beginning
    centers[:,1] = X[:,z0]
    for j = 2:k
        c, a = max_dist(X,centers[:,1:j-1])
        centers[:,j] = X[:,a]
    end
    return centers
end
=#

function fft(X,k)
    d,n = size(X)
    centers = zeros(d,k)
    z0 = rand(1:n,1)  # randomly select a sample as the beginning
    centers[:,1] = X[:,z0]
    dist = Array{Float64}(undef, n)
    dist_matrix = Array{Float64}(undef, n, 1)
    for j = 2:k
        if j == 2
            dist = view(pairwise(SqEuclidean(),X, centers[:,(j-1):(j-1)], dims=2),:,1)
        else
            pairwise!(dist_matrix, SqEuclidean(), X, centers[:,(j-1):(j-1)], dims=2)    
            dist = min.(dist, dist_matrix[:,1])
        end
        c, a = findmax(dist)
        centers[:,j] = X[:,a]
    end
    return centers
end


function fft_FBBT(X, k, lower, upper)
    d,n = size(X)
    centers = zeros(d,k+1)
    centers[:,1] = (lower+upper)/2
    dist = Array{Float64}(undef, n)
    dist_matrix = Array{Float64}(undef, n, 1)
    for j = 2:(k+1)
        if j == 2
           dist = view(pairwise(SqEuclidean(),X, centers[:,(j-1):(j-1)], dims=2),:,1)
        else
           pairwise!(dist_matrix, SqEuclidean(), X, centers[:,(j-1):(j-1)], dims=2)
           dist = min.(dist, dist_matrix[:,1])
        end
        c, a = findmax(dist)
        centers[:,j] = X[:, a]
    end
    return centers[:,2:(k+1)]
end




function getUpperBound(X, k, lower, upper, tol = 0)
    UB = Inf
    centers = nothing
    for tr = 1:100
        t_ctr = fft(X,k)
        t_UB = kcenter_opt.obj_assign(t_ctr, X)
        if tol <= UB - t_UB
            UB = t_UB
            centers = t_ctr
        end
    end

    ##### get upper bound from random centers
    d,n = size(X)
    t_centers = (lower.+upper)/2
    t_ctr = kcenter_opt.sel_closest_centers(t_centers, X)
    for tr = 1:100
        t_UB = kcenter_opt.obj_assign(t_ctr, X)
        if tol <= UB - t_UB
            UB = t_UB
            centers = t_ctr
        end
        inds = sample(1:n, k, replace=false)
        t_ctr = X[:, inds]
        #t_centers = rand(d, k).*(upper - lower) .+ lower      
        #@time t_ctr = kcenter_opt.sel_closest_centers(t_centers, X) 
    end
    return centers, UB
end

# end of the module
end