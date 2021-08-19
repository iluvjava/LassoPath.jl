mutable struct LassoSCOP <: LassoRoot
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

    # Paramters for the iterator
    Tol::Float64
    λMin::Float64

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
        this = new(A, y, μ, u, Z, l, OptModel, λ)
        this.Tol = 1e-8
        this.λMin = 1e-8
        return this
    end
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
        Solve for the weights of the current model, 
        given the current configuration of the model.
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