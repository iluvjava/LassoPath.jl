
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