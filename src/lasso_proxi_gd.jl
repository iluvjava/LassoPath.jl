### ============================================================================
### A type related to Proximal Gradient and Lasso
### ============================================================================

mutable struct ProximalGradient
    """
        Dynamical, functional, keeps references to detailed implementation 
        functions. 
    """

    f::Function 
    gradient::Function
    prox::Function
    β::Float64  # For beta convex function. 

    function ProximalGradient()
        return new()
    end

end


function L1LassoProximal(y::Vector, t::Float64, λ::Float64)
    """
        Textbook definition of the L1 Lasso proximal operator. 
    """

    return map(y) do yi
        if yi >= λ*t
            return yi - λ*t
        else if yi <= -λ*t
            return yi + λ*t
        else
            return 0
        end
    end

end


function BuildProximalGradientLasso(
        A::Matrix, 
        b::Matrix, 
        λ::Float64
    )::ProximalGradient
    """
        Given matrix A, vector b, and λ the regularization perameter, 
        this will build an instance of ProximalGradient. 

        * Use Lambdas to capture variables references 
        * A factory methods
        ! Pass copy of matrices will be better. 
    """
    # TODO: Test this 

    @assert b.dims == 2 && length(b) == size(b, 1) "Expect vector b in the"*
    " shape of (n, 1) but it turns out to be $(size(b))"
    @assert size(b, 1) == size(A, 2) "Expect the size of the matrix to match "*
    "the vector but it turns out to be A is in $(size(A)), and b is $(size(b))"

    β = 4*opnorm(A)  # convexity from spectral norm. 
    t = 1/β
    f(x) = norm(A*x - b)^2
    df(x) = A'*(A*x - b)
    
    # ======= build ==================================
    proxVectorized(y) = L1LassoProximal(y, t, λ)
    this = ProximalGradient()
    this.f = f
    this.gradient = df
    this.prox = proxVectorized
    this.β = β
    return this
end


function OptimizeProximalGradient(
        this::ProximalGradient, 
        warmstart::Matrix, 
        tol::Float64=1e-8
    )::Matrix
    """
        Implement FISTA, Accelerated Proximal Gradient. 
    """
    x = warmstart
    Δx = Inf
    y = x
    t = 1
    
    # TODO: Implement this
    while Δx >= tol
        
    end

    return x

end


### ----------------------------------------------------------------------------

### ============================================================================
### A type related to doing lass, using the proximal gradient method
### ============================================================================


mutable struct LassoProximal <: LassoRoot
    ## It's just a collection of data. 

    A::Matrix # Feature matrix 
    y::Matrix # label vector 
    μ::Matrix # feature mean
    u::Number # label mean
    Z::Matrix # Starndardized Matrix
    l::Matrix # Zero mean label 
    
    OptModel::ProximalGradient  # The package of ProximalGradient method stuff
    λ::Float64       # the regularization parameter. 

    LassoPath::Union{Matrix, Nothing} # going along a fixed row is the fixing 
    # the feature while varying the lambda quantity. 
    λs::Union{Vector, Nothing} # the lambda values. 

    # Paramters for the iterator
    Tol::Float64
    λMin::Float64

    function LassoLassoProximal(A::Matrix, y::Union{Matrix, Vector}, λ::Float64=0.0)
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
        OptModel = BuildProximalGradientLasso(Z, l, λ)
        this = new(A, y, μ, u, Z, l, OptModel, λ)
        this.Tol = 1e-8
        this.λMin = 1e-8
        return this
    end
end




function Changeλ(this::LassoProximal, λ::Float64)
    """
        Change the Lasso regularizer of the current model 
    """
    # TODO: Implement this 
    
end


function SolveForx(this::LassoProximal)
    """
        Solve for the weights of the current model, 
        given the current configuration of the model.
    """

    # TODO: Implement this 
    
end

