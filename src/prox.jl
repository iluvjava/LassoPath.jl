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


function OptimizeProximalGradient(
        this::ProximalGradient, 
        warm_start::Union{Array, Nothing}=nothing
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
        yNew .= xNew
        yNew += ((t - 1)/tNew)*(xNew - x)

        ∇f .= this.gradient(yNew)
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
### Lasso Proximal Operator
### ============================================================================

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


### ----------------------------------------------------------------------------

### ============================================================================
### The 2 Norm loss function with L1 regularization
### ============================================================================


function BuildPG2NormLasso(
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
    ATb = A'*b
    β = 4*opnorm(ATA)  # convexity from spectral norm. 
    t = 1/β
    f(x) = norm(A*x - b)^2
    dg(x) = 2(ATA*x - ATb)
    
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
    """
        Change the proximal operator for a different λ
    """
    this.prox = (y, t) -> L1LassoProximal!(y, t, λ)
end

# ------------------------------------------------------------------------------

### ============================================================================
### Lasso huber loss function and l1 norm regularization
### ============================================================================


function BuildPGHuberLasso(
    A::Matrix,
    B::Array,
    a::Float64,
    δ::Float64,
    λ::Float64
)
    # TODO: implement this 


end

### ============================================================================
### Lasso Cross Entropy loss function and l1 norm regularization
### ============================================================================

function BuildPGEntropyLasso()

end
