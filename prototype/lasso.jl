

using COSMO, JuMP, LinearAlgebra
using Statistics
using JuMP
import Plots as Plots
import ProgressMeter as Pm
MOI = JuMP.MathOptInterface

include("utils.jl")

function MakeLassoOptimizationProblem(A::Matrix, y::Matrix, λ::Float64)
    """
        Phrase the quadratic programming problem for Lasso regularization 
        problem. 
        
    """
    m, n = size(A)
    @assert size(y, 1) == m && size(y, 2) == 1 "the label and"* 
    " the data matrix doesn't have the maching dimension X:" * 
    string(m, "×",  n) * string(";y: ", size(y))
    @assert λ ≤ 1 && λ ≥ 0 "Regularization λ should be between (0, 1)"


    Pb = Pm.ProgressUnknown("Building Optimization Model: ", spinner=true)

    model = Model(with_optimizer(COSMO.Optimizer); bridge_constraints = false)

    set_optimizer_attribute(model, MOI.Silent(), true)
    set_optimizer_attribute(model, "max_iter", 5000*10)

    @variable(model, x[1:n]); Pm.next!(Pb)
    setvalue.(x, A\y); Pm.next!(Pb)
    @variable(model, η[1:n]); Pm.next!(Pb)
    @constraint(model, -η .<= x); Pm.next!(Pb)
    @constraint(model, x .<= η); Pm.next!(Pb)
    @objective(model, Min, λ*sum(η) + sum((A*x - y).^2)); Pm.finish!(Pb)

    return model

end


mutable struct LassoSCOP
    ## It's just a collection of data. 

    A::Matrix # Feature matrix
    y::Matrix # label vector 
    μ::Matrix # feature mean
    u::Number # label mean
    Z::Matrix # Starndardized Matrix
    l::Matrix # Zero mean label 
    
    OptModel::Model  # The JuMP model for getting it right. 
    λ::Float64       # the regularization parameter. 

    LassoPath::Union{Matrix, Nothing} # going along a fixed row is the fixing 
    # the feature while varying the lambda quantity. 
    λs::Union{Vector, Nothing} # the lambda values. 

    function LassoSCOP(A::Matrix, y::Union{Matrix, Vector}, λ::Float64=0.0)
        A = copy(A)
        m, _ = size(A)
        @assert size(y, 1) == m ""*
        "The rows of X should match of the columns of y, but the size of"*
        string("X, Y is: ", size(A), " ", size(y))
        
        y = copy(y)
        if ndims(y) == 1
            y = reshape(y, (length(y), 1))
        end
        μ = mean(A, dims=1)::Matrix
        u = mean(y)
        Z = A .- μ
        l = y .- u
        OptModel = MakeLassoOptimizationProblem(Z, l, λ)

        return new(A, y, μ, u, Z, l, OptModel, λ)
    end
end


function LassoPath(this::LassoSCOP, tol::Float64=1e-8)
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
    Changeλ(this, λ)
    x = SolveForx(this)
    dx = Inf
    push!(Results, x)
    push!(λs, λ)
    MaxItr = 100
    pb = Pm.ProgressThresh(tol, "inf norm of δx: ")
    while dx >= tol && MaxItr >= 0
        push!(λs, λ)
        λ /= 2
        MaxItr -= 1
        setvalue.(this.OptModel[:x], value.(x)) # warm start! 
        Changeλ(this, λ)
        x = SolveForx(this)
        push!(Results, value.(x))
        dx = norm(Results[end - 1] - Results[end], Inf)
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


function VisualizeLassoPath(this::LassoSCOP, 
                            fname::Union{String, Nothing}=nothing,
                            title::Union{String, Nothing}=nothing
                            )
    """
        Make a plots for the lasso path and save it in the pwd. 
        
    """
    @assert isdefined(this, :LassoPath) "Lasso Path not defined for the object"*
    "yet". 
    error("Haven't implemented it yet.")
    λs = this.λs
    Paths = this.LassoPath
    Loggedλ = log2.(λs)
    p1 = Plots.heatmap(Paths[end:-1:begin, :], title=title===nothing ? "Lasso Path" : title)
    p2 = plot(Loggedλ, Paths', label=nothing)
    Plots.xlabel!(p2, "log_2(λ)")
    p = Plots.plot(p1, p2, layout=(2, 1))
    Plots.plot!(size=(600, 800))
    Plots.plot!(dpi=400)
    Plots.savefig(p, fname===nothing ? "plot.png" : fname)
    return
end


function CaptureImportantWeights(
                                this::LassoSCOP, 
                                top_k::Union{Float64, Int64} = 0.5, 
                                threshold::Float64=1e-10
                                )
    """
        Caputre the indices for the most important predictors from the 
        regression. Returns the indices of the important weights. 
        
        ---
        **this::LassoSCOP**: 
            An instance of the type LassoSCOP
        
    """
    # TODO: test this 
    @assert isdefined(this, :LassoPath) "Lasso Path not defined for this"*
    "Object yet. "
    @assert top_k >= 0 "this parameters, should be a positive number"
    @assert threshold >= 0 "This parameters shouold be a positive number"

    Paths = this.LassoPath
    top_k::Int64 = top_k <= 1 ? ceil(size(Paths, 1)*0.5) : round(top_k)
    for JJ in size(Paths, 2)
        Col = view(Paths, :, JJ)
        NonNegative = sum(abs.(Col) .>= threshold)
        # rank them by abs and returns the indices for top k weights
        if NonNegative >= top_k 
            Indices = sortperm(abs.(Col), rev=true)
            return Indices[begin:top_k]
        end
    end
    return Vector{Int64}()
end


function Changeλ(this::LassoSCOP, λ)
    """
        Change the Lasso regularizer of the current model 
    """
    model = this.OptModel
    x = model[:x]  # objects can be indexed with symbols! 
    η = model[:η]
    y = this.l
    A = this.Z
    @objective(model, Min, λ*sum(η) + sum((A*x - y).^2))
    
end


function SolveForx(this::LassoSCOP)
    """
        Solve for the weights of the current model, given the current configuration of the model.
    """
    optimize!(this.OptModel)
    TernimationStatus = termination_status(this.OptModel)
    # @assert TernimationStatus == MOI.OPTIMAL "Terminated with non-optimal value when solving for x. "*
    # string("The status is: ", TernimationStatus)*"\n this is the results \n $(OptResults)"
    
    if !(TernimationStatus == MOI.OPTIMAL)
        Warn("\nWarning: convergence status for solver: $(TernimationStatus)")
        Warn("Current Value λ: $(this.λ)")
        PrintTitle("Here is the summary for the solution: ")
        display(solution_summary(this.OptModel))
    end
    return value.(this.OptModel[:x])
end


function Getαβ(this::LassoSCOP, lambda::Float64)
    """
        Get the weights and biases, for the original model (The model trained is normalized), for a 
        particular regularization value. 
    """
    error("Not yet implemented. ")
    
end


# TODO: Override Base.show for this LASSOPath TYPE.
