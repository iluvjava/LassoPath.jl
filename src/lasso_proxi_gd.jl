### ============================================================================
### A type related to Proximal Gradient and Lasso
### ============================================================================

mutable struct ProximalGradient
    """
        Dynamical, functional, keeps references to detailed implementation 
        functions. 
    """

    g::Function 
    gradient::Function
    prox::Function
    h::Function
    
    β::Float64  # For beta convex function. 
    tol::Float64
    maxItr::Int64 
    solutionDim::Tuple

    function ProximalGradient(tol::Float64, maxItr::Int64)
        this = new()
        this.tol = tol
        this.maxItr = maxItr
        return this
    end

end


function L1LassoProximal!(y::Union{Vector, Matrix}, t::Float64, λ::Float64)
    """
        Textbook definition of the L1 Lasso proximal operator. 
    """

    return map!(y, y) do yi
        if yi > λ*t
            return yi - λ*t
        elseif yi < -λ*t
            return yi + λ*t
        else
            return 0
        end
    end

end


function BuildProximalGradientLasso(
        A::Matrix, 
        b::Matrix, 
        λ::Float64, 
        tol::Float64=1e-5,
        max_itr::Int64=1000
    )::ProximalGradient
    """
        Given matrix A, vector b, and λ the regularization perameter, 
        this will build an instance of ProximalGradient. 

        * Use Lambdas to capture variables references 
        * A factory methods
        ! Pass copy of matrices will be better. 
    """

    @assert size(b, 2) == 1 "Expect vector b in the"*
    " shape of (n, 1) but it turns out to be $(size(b))"
    @assert size(b, 1) == size(A, 1) "Expect the size of the matrix to match "*
    "the vector but it turns out to be A is in $(size(A)), and b is $(size(b))"
    ATA = A'*A
    β = 4*opnorm(ATA)  # convexity from spectral norm. 
    t = 1/β
    f(x) = norm(A*x - b)^2
    dg(x) = 2(ATA*x - A'*b)
    
    # ======= build ==================================
    proxVectorized(y, t) = L1LassoProximal!(y, t, λ)
    this = ProximalGradient(tol, max_itr)
    this.g = f
    this.gradient = dg
    this.prox = proxVectorized
    this.β = β
    this.solutionDim = (size(A, 2), size(b, 2))
    this.h = (x) -> norm(x, 1)

    return this
end

function ChangeProximalGradientLassoλ!(this::ProximalGradient, λ::Float64)
    this.prox = (y, t) -> L1LassoProximal!(y, t, λ)
end


function OptimizeProximalGradient(
        this::ProximalGradient, 
        warm_start::Union{Matrix, Nothing}=nothing
    )::Matrix
    """
        Implement FISTA, Accelerated Proximal Gradient, copied from my HW. 
    """
    tol = this.tol
    max_itr = this.maxItr
    
    if warm_start === nothing
        warm_start = zeros(this.solutionDim)
    end

    x  = warm_start
    Δx = Inf
    y  = x
    t  = 1
    ∇f = this.gradient(y)
    δ  = 1/this.β # stepsize
    xNew = similar(x)
    yNew = similar(y)

    while Δx >= tol && max_itr >= 1
        xNew = x - δ*∇f
        this.prox(xNew, δ)
        tNew = (1 + sqrt(1 + 4t^2))/2
        yNew = xNew + ((t - 1)/tNew)*(xNew - x)

        ∇f = this.gradient(yNew)
        Δx = norm(xNew - x, 1)/norm(x, 1)
    
        t = tNew
        x = xNew
        y = yNew
        
        max_itr -= 1
    end
    if max_itr == 0
        println()
        Warn("Maximal iteration $(this.maxItr) reached for Proximal gradient. ")
        @assert norm(x, Inf) != Inf "Solution is infinite, something blowed up."
        Warn("It will be bumped up by 200% more and then we try again.")
        this.maxItr = 2*this.maxItr
        return OptimizeProximalGradient(this, warm_start)
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
    y::Matrix # label vector/matrix 
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

    function LassoProximal(
            A::Matrix, 
            y::Union{Matrix, Vector}, 
            λ::Float64=0.0
        )
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
        this.Tol = 1e-4
        this.λMin = 1e-8
        return this
    end
end


function Changeλ!(this::LassoProximal, λ::Float64)
    """
        Change the Lasso regularizer of the current model 
    """

    ChangeProximalGradientLassoλ!(this.OptModel, λ)
    return 
end


function SolveLasso(
            this::LassoProximal; 
            x0::T=nothing
        )::Vector{Float64} where {T <: Union{Vector{Float64}, Nothing}}
    """
        Solve for the weights of the current model, 
        given the current configuration of the model.
    """

    if !(x0 === nothing)
        x0 = reshape(x0, this.OptModel.solutionDim)
    end

    return reshape(OptimizeProximalGradient(this.OptModel, x0), :)
    
end

