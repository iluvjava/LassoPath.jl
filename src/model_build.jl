function MakeLassoOptimizationProblem(
    A::Matrix{Float64}, 
    y::Matrix{Float64}, 
    λ::Float64
    )
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


function MakeQuantileLassoOptimizationProblem(
        A::Matrix{Float64}, 
        y::Matrix{Float64}, 
        λ::Float64=0, 
        q::Float64=0.5,
    )
    """
        Phrase regularized, quantile regression problem as a linear programming problem. 
        problem. 
        
    """
    @assert size(A, 2)  == length(y) "The number of columns for matrix A"*
    " should equals to the number of elements in vector b. "
    @assert λ >= 0 "The L1 regularization parameter should be positive"
    @assert q <= 1 && q >= 0 "The quantile parameter is not percentile"*
    "It should be in between 0 and 1. "

    

end