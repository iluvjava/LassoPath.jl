### ============================================================================
### Method related to lasso path associated with type `LassoRoot`
### ============================================================================

"""
    Get the lassopath for the instance made, the function will modify the 
    λs field, and the LassoPath field of the instance. 
    
"""
function GetLassoPath(this::LassoRoot, decay_by=0.9)
    """
        Analyze the Lasso Problem by drawing a lasso path. It will start with 
        a parameter that will make all predictors zero and then solve it 
        iterative by chopping the regularization λ by half each iteration. 


    """
    @assert decay_by != 1 "lambda decay value cannot be 1. "
    if decay_by < 1 
        decay_by = 1/decay_by
    end

    u = this.u
    A = this.Z
    y = this.l
    Results = Vector{Vector}()
    
    function λMax(A, y)
        ToMax = A'*(y .- u)
        ToMax *= 2
        return maximum(abs.(ToMax))
    end

    λ = λMax(A, y)
    λs = Vector{Float64}()
    Changeλ!(this, λ)
    x = SolveLasso(this)
    dx = Inf
    push!(Results, x)
    push!(λs, λ)
    pb = Pm.ProgressThresh(this.Tol, "inf norm of δx: ")
    while dx >= this.Tol && λ >= this.λMin
        λ /= decay_by
        push!(λs, λ)
        Changeλ!(this, λ)
        x = SolveLasso(this, x0=x)
        push!(Results, value.(x))
        dx = norm(Results[end - 1] - Results[end], 1)/norm(Results[end - 1], 1)
        Pm.update!(pb, dx)
    end


    ResultsMatrix = zeros(length(x), length(Results))
    for II ∈ 1:length(Results)
        ResultsMatrix[:, II] = Results[II]
    end

    # Store it. 
    this.LassoPath = ResultsMatrix
    this.λs = λs
    return ResultsMatrix, λs
end


function VisualizeLassoPath(
        this::LassoRoot;
        fname::Union{String, Nothing}=nothing,
        title::Union{String, Nothing}=nothing
    )
    """
        Make a plots for the lasso path and save it in the pwd. 
        
    """
    @assert isdefined(this, :LassoPath) "Lasso Path not defined for the object"*
    "yet, use GetLassoPath() function on it first". 
    error("Haven't implemented it yet.")
    λs = this.λs
    Paths = this.LassoPath
    Loggedλ = log2.(λs)
    p1 = Plots.heatmap(
            Paths[end:-1:begin, :], 
            title=title===nothing ? "Lasso Path" : title
        )
    p2 = Plots.plot(Loggedλ, Paths', label=nothing)
    Plots.xlabel!(p2, "log_2(λ)")
    p = Plots.plot(p1, p2, layout=(2, 1))
    Plots.plot!(size=(600, 800))
    Plots.plot!(dpi=400)
    Plots.savefig(p, fname===nothing ? "plot.png" : fname)
    return
end

"""
    Caputre the indices for the most important predictors from the 
    regression. Returns the indices of the important weights. 

    It decreases the λ geometrically and count the number of non-zero 
    weigthts at for each λ. Once the number of non-zero weights reached 
    the threshold (top_k), it will return the indices of these non-zero 
    weights. 
        
    ---
    `this::LassoRoot`: 
        An instance of the Lasso Analyzer. 
    
    `top_k::Union{Float64, Int64}`:
        The number of non-zero weights you want, if it's a float between 0, 1, then 
        it's interpreted as the ratio of non-zero weights you want. 
    
    `threshold::Float64=1e-4`
        Set a threshold for the weights that you deem to be "important", only 
        weights larger than this threshold will be viiwed as "non-zero". 

    ---
    `returns`: 
        * indices of the important predictors
        * values of the regularization parameters
        * the actual weigths
"""
function CaptureImportantWeights(
        this::LassoRoot, 
        top_k::Union{Float64, Int64} = 0.5, 
        threshold::Float64=1e-10
    )
    
    @assert isdefined(this, :LassoPath) "Lasso Path not defined for this"*
    "Object yet. "
    @assert top_k >= 0 "this parameters, should be a positive number"
    @assert threshold >= 0 "This parameters shouold be a positive number"

    Paths = this.LassoPath
    top_k::Int64 = top_k <= 1 ? ceil(size(Paths, 1)*0.5) : round(top_k)
    for JJ in 1:size(Paths, 2)
        Col = view(Paths, :, JJ)
        BigEnough = sum(abs.(Col) .>= threshold)
        # rank them by abs and returns the indices for top k weights
        if BigEnough >= top_k 
            Indices = sortperm(abs.(Col), rev=true)
            Indices = Indices[begin:top_k]
            return Indices, this.λs[JJ], this.LassoPath[Indices, JJ]
        end
    end
    return Vector{Int64}()
end

"""
    Find the weights that always monotinically increases as the value of 
    λ decreases. 
    
    Monotone weights are likely to be independent and have less covariances 
    with other predictors. 
"""
function CaptureMonotoneWeights(this::LassoRoot)
    # TODO: Implement his
    
end


function WeightCorrelations(this::LassoRoot, index::Int64)
    # TODO: implement this
end


function λSuchThatThisWeightIsZero(this::LassoRoot)
    # TODO: implement this

end


function RankAllWeights(this::LassoRoot)
    # TODO: Implement This    

end

