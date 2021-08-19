# Analyze the data using Lasso Path pacakge I wrote. 
cd("C:\\Users\\victo\\source\\repos\\Silly_Julia_stuff\\Julia Packages\\LassoPath\\usage_examples\\crime_data")
using DelimitedFiles, CSV, DataFrames
using LassoPath

Df = CSV.read("process_data.csv", DataFrame)
Df = Df[!, Not([:state, :fold])]

# Extrac Numerics, predictors, predictants. 
A, b = Matrix(Df[:, Not(end)]), Df[:, end]

PrintTitle("The Data is ready, we are ready for analyais now")

lassoIntance = LassoSCOP(A, b)

GetLassoPath(lassoIntance)

VisualizeLassoPath(lassoIntance)

predictorIndices, λ, ws = CaptureImportantWeights(lassoIntance, 10, 1e-2)

PrintTitle("Important Predictors are")

predictors = names(Df[:, Not(end)])[predictorIndices]
impacts = map(ws) do x
    if x >= 0 
        return "+"
    else 
        return "-"
    end
end

display([reshape(predictors, :) reshape(impacts, :)])
