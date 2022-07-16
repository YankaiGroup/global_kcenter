module branch

using Distributions: maximum
using LinearAlgebra, Distributions, RDatasets
using Nodes
using kcenter_lb

export branch!



function SelectVarMaxRange(node)
    dif = node.upper -node.lower
    ind = findmax(dif)[2]
    return ind[1], ind[2]
end

function branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k, bt_method, symmtrc_breaking)
    d, n = size(X);
    lower = copy(node.lower)
    upper = copy(node.upper)
    upper[bVarIdx, bVarIdy] = bValue # split from this variable at bValue
    if symmtrc_breaking ==1
        for j = 1:(k-1)  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
    	    if upper[1, k-j] >= upper[1, k-j+1]  
	            upper[1, k-j] = upper[1, k-j+1]
            end
	end
    end    
    if sum(lower.<=upper)==d*k
    	left_node = Node(lower, upper, node.level+1, node_LB, copy(node.assign), copy(node.center_cand))
	push!(nodeList, left_node)
	# println("left_node:   ", lower, "   ",upper)
    end

    lower = copy(node.lower)
    upper = copy(node.upper)
    lower[bVarIdx, bVarIdy] = bValue
    if symmtrc_breaking ==1
        for j = 2:k
    	    if lower[1, j] <= lower[1, j-1]
	            lower[1, j]	= lower[1, j-1]
            end
	end
    end
    if sum(lower.<=upper)==d*k
    	right_node = Node(lower, upper, node.level+1, node_LB, copy(node.assign), copy(node.center_cand))
    	push!(nodeList, right_node)
	#println("right_node:   ", lower,"   ",upper)
    end
end

end
