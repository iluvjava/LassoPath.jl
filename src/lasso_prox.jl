
### ============================================================================
### A type related to doing lass, using the proximal gradient method for L2
### regression 
### ============================================================================


mutable struct Lasso2NormProx <: LassoRoot
    ## It's just a collection of data. 

    A::Matrix # Feature matrix 
    y::Matrix # label vector/matrix 
    μ::Matrix # feature mean
    u::Number # label mean
    Z::Matrix # Starndardized Matrix
    l::Matrix # Zero mean label 
    
    proxOptim::ProximalGradient  # The package of ProximalGradient method stuff
    λ::Float64       # the regularization parameter. 

    lassoPath::Union{Matrix, Nothing} # going along a fixed row is the fixing 
    # the feature while varying the lambda quantity. 
    λs::Union{Vector, Nothing} # the lambda values. 

    # Paramters for the iterator
    tol::Float64
    λmin::Float64

    function Lasso2NormProx(
            A::Matrix, 
            y::Array, 
            λ::Float64=0.0
        )
        warn("This type has been deprecated due to performance issues. ")
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
        OptModel = BuildPG2NormLasso(Z, l, λ)
        this = new(A, y, μ, u, Z, l, OptModel, λ)
        this.Tol = -1
        this.λMin = 1e-8
        return this
    end
end


function Changeλ!(this::Lasso2NormProx, λ::Float64)
    """
        Change the Lasso regularizer of the current model 
    """

    ChangeProximalGradientLassoλ!(this.proxOptim, λ)
    return 
end


function SolveLasso(
            this::Lasso2NormProx; 
            x0::T=nothing
        )::Vector{Float64} where {T <: Union{Vector{Float64}, Nothing}}
    """
        Solve for the weights of the current model, 
        given the current configuration of the model.
    """

    if !(x0 === nothing)
        x0 = reshape(x0, this.proxOptim.solutionDim)
    end

    return reshape(OptimizeProximalGradient(this.proxOptim, x0), :)
    
end

