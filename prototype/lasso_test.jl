include("lasso.jl")
include("utils.jl")

using Test
using UnicodePlots
using Plots

Test1Results = 0

function Test1()
    """
        Test 1, some basic test on a random matrix, copied columns of matrix. 
    """
    deg = 3
    N = 2000
    B = rand(N, deg)
    A = fill(0.0, (N, 2deg))
    A[:, 1:deg] = B
    A[:, deg + 1:end] = 2B.^2 + B + 0.1.*randn(size(B))

    b = A*ones(2deg, 1)
    instance = LassoSCOP(A, b)
    results, λs = LassoPath(instance)
    display(results)
    display(λs)
    
    # Let's plot this
    global Test1Results = results

    ylim = (minimum(Test1Results), maximum(Test1Results))
    ThePlot = lineplot(log2.(λs), Test1Results[1, :], ylim=ylim, title="Lasso Path")

    for II ∈ 2:size(Test1Results, 1)
        lineplot!(ThePlot, log2.(λs), Test1Results[II, :])
    end
    display(ThePlot)
    ThePlot = UnicodePlots.heatmap(results[end:-1:begin, :])
    display(ThePlot)

    # Visualize the path. 
    VisualizeLassoPath(instance)

    # Finding the important weights for it. 
    PrintTitle("These are the important weights: ")
    WeightsIndices = CaptureImportantWeights(instance)
    println(WeightsIndices)
    return true

end

@test Test1()


function Test2()
    N = 10
    deg = 10
    xgrid = collect(LinRange(-2π, 2π, N))
    f(x) = sin.(x)
    y = f(xgrid) + 0.1.*rand(N)
    V = VanderMonde(reshape(xgrid, (N, 1)), deg)
    instance = LassoSCOP(V, reshape(y, (N, 1)))
    results, λs = LassoPath(instance)
    λs = log2.(λs)
    display(results)

    ylim = (minimum(results), maximum(results))
    ThePlot = lineplot(λs, results[1, :], ylim=ylim)
    for II ∈ 2:size(results, 1)
        lineplot!(ThePlot, λs ,results[II, :])
    end
    display(ThePlot)

    YHat = V*results[:, end]
    ThePlot = lineplot(xgrid, y)
    scatterplot!(ThePlot, xgrid, YHat)
    display(ThePlot)
    display(UnicodePlots.heatmap(results[end:-1:begin, :]))
    
    return true
end

@test Test2()