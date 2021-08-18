### Use the lasso path for the crime train example. 

include("lasso.jl")
include("utils.jl")

using DataFrames
using DelimitedFiles: readdlm
using Plots
import UnicodePlots as Uplot


function DataSummary() 
    P, H = readdlm("./L1_Lasso_Module/crime-train.txt", '\t'; header=true)
    Df = DataFrame(P, reshape(H, length(H)))
    return Df
end


PrintTitle("A Summarization of the Data: The Predictors")
TheData= DataSummary()
println(describe(TheData))
println("The size of the dataframe is: $(size(TheData))")
println("The predictant is:\"$(names(TheData)[1])\"")
PrintTitle("We are gonna prepare the into a matrix and vector. ")

function PrepareTheDataMatrix(data_frame::DataFrame) 
    """
        Min, max, median and mean are assumed to be important predictor
    """
    FeatureMatrix = Matrix(data_frame[!, 2:end])
    Predictants = data_frame[!, 1]
    return FeatureMatrix, Predictants
end

A, b = PrepareTheDataMatrix(TheData)


function AnalyaisWithLasso(A::Matrix, b::Matrix)
    """
        Do the lasso path thing on the data matrix, 
        features are columns of the matrix, instances are rows. 

        Returns: 
            tuple where the first element is the data matrix and the 
            second element is the column vector of all the regularization
            parameter λ. 
    """
    Instance = LassoSCOP(A, b)
    Results, λs =  LassoPath(Instance)
    VisualizeLassoPath(Instance, "cirme-data-lassopath-$(rand(1:999999999)).png")
    

end

AnalyaisWithLasso(A, reshape(b, (length(b), 1))) 