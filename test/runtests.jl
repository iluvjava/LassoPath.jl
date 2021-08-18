using LassoPath
using Test
using UnicodePlots
using Plots


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
    results, λs = GetLassoPath(instance)
    display(results)
    display(λs)
    
    # Let's plot this, and a global variable will be defined. 
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


function Test2()

    PrintTitle("Testing the Iterator for the Lasso Path type. ")
    deg = 3
    N = 2000
    B = rand(N, deg)
    A = fill(0.0, (N, 2deg))
    A[:, 1:deg] = B
    A[:, deg + 1:end] = 2B.^2 + B + 0.1.*randn(size(B))
    
    # Use this to test the lasso iterate. 
    let A = A, b = A*ones(2deg, 1)
        l = LassoSCOP(A, b )
        for (λ, x) ∈ l
            println(λ)
            println(x)
        end
    end

    return true
end


# A set of basic tests for Lasso. 
@testset "LassoPath.jl" begin
    PrintTitle("Test Set 1")
    @test Test1()
    @test Test2()
        

    
end
