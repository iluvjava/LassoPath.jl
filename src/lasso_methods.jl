### ============================================================================
### Method related to lasso path associated with type `LassoRoot`
### ============================================================================

"""
    Get the lassopath for the instance made, the function will modify the 
    λs field, and the LassoPath field of the instance. 
    
"""
function GetLassoPath(this::LassoRoot)
    """
        Analyze the Lasso Problem by drawing a lasso path. It will start with 
        a parameter that will make all predictors zero and then solve it 
        iterative by chopping the regularization λ by half each iteration. 

        **this**: 
            An instance of the LassoSCOP
        **tol**: 
            If the infinity norm of vector of the change in the weights is 
            less than this quantity, then it stops and return all the results. 

    """

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
    MaxItr = 100
    pb = Pm.ProgressThresh(this.Tol, "inf norm of δx: ")
    while dx >= this.Tol && MaxItr >= 0
        λ /= 2
        push!(λs, λ)
        MaxItr -= 1
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


function CaptureImportantWeights(
        this::LassoRoot, 
        top_k::Union{Float64, Int64} = 0.5, 
        threshold::Float64=1e-10
    )
    """
        Caputre the indices for the most important predictors from the 
        regression. Returns the indices of the important weights. 
        
        ---
        **this::LassoSCOP**: 
            An instance of the type LassoSCOP

        ---
        returns: 
            * indices of the important predictors
            * values of the regularization parameters
            * the actual weigths
    """
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


function CaptureMonotoneWeights(this::LassoRoot)
    """
        Monotonically increasing weights (in absolute value) as the value of 
        λ increases, are the weights that are not important, but also doesn't 
        correlates to each other. 
    """
    # TODO: Implement his

end


function RecoverModelWeights(this::LassoRoot)
    """
        Recovers the weights for the model, because the LassoRoot is regularized
        only with normalized data. 
    """
    # TODO: Implement this 
end



